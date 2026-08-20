//! The TCP rendezvous and control wire encoding.
//!
//! This module is gated at the declaration site in `mod.rs`, so its items do
//! not repeat the `std` feature attribute individually.

use std::io::{Read, Write};
use std::net::TcpStream;

use super::error::{ContextError, network};
use super::identity::RunId;
use super::state::ContextFailure;
use super::{MAX_RUN_ID_BYTES, TWO_RANK_WORLD};

const PROTOCOL_MAGIC: [u8; 8] = *b"INCINRV1";
const STARTUP_BYTES: usize = 160;
const CONTROL_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum MessageKind {
    Hello = 1,
    Accepted = 2,
    Reject = 3,
    Shutdown = 4,
    Abort = 5,
}

impl MessageKind {
    fn decode(value: u8) -> Result<Self, ContextError> {
        match value {
            1 => Ok(Self::Hello),
            2 => Ok(Self::Accepted),
            3 => Ok(Self::Reject),
            4 => Ok(Self::Shutdown),
            5 => Ok(Self::Abort),
            _ => Err(ContextError::Protocol("unknown rendezvous message kind")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct StartupMessage {
    pub(super) kind: MessageKind,
    rank: u8,
    world: u8,
    pub(super) code: u16,
    pub(super) local_cuda_device: u64,
    run_id_len: u16,
    run_id: [u8; MAX_RUN_ID_BYTES],
}

impl StartupMessage {
    pub(super) fn hello(
        run_id: &RunId,
        rank: usize,
        local_cuda_device: usize,
    ) -> Result<Self, ContextError> {
        Self::new(MessageKind::Hello, run_id, rank, local_cuda_device)
    }

    pub(super) fn accepted(
        run_id: &RunId,
        rank: usize,
        local_cuda_device: usize,
    ) -> Result<Self, ContextError> {
        Self::new(MessageKind::Accepted, run_id, rank, local_cuda_device)
    }

    fn new(
        kind: MessageKind,
        run_id: &RunId,
        rank: usize,
        local_cuda_device: usize,
    ) -> Result<Self, ContextError> {
        let mut bytes = [0; MAX_RUN_ID_BYTES];
        bytes[..run_id.0.len()].copy_from_slice(run_id.0.as_bytes());
        Ok(Self {
            kind,
            rank: u8::try_from(rank)
                .map_err(|_| ContextError::RankOutOfRange { rank, world: 2 })?,
            world: TWO_RANK_WORLD as u8,
            code: 0,
            local_cuda_device: u64::try_from(local_cuda_device).map_err(|_| {
                ContextError::Protocol("local CUDA ordinal does not fit rendezvous wire")
            })?,
            run_id_len: run_id.0.len() as u16,
            run_id: bytes,
        })
    }

    pub(super) const fn reject(code: u16) -> Self {
        Self {
            kind: MessageKind::Reject,
            rank: 0,
            world: TWO_RANK_WORLD as u8,
            code,
            local_cuda_device: 0,
            run_id_len: 0,
            run_id: [0; MAX_RUN_ID_BYTES],
        }
    }

    fn encode(self) -> [u8; STARTUP_BYTES] {
        let mut bytes = [0; STARTUP_BYTES];
        bytes[..8].copy_from_slice(&PROTOCOL_MAGIC);
        bytes[8] = self.kind as u8;
        bytes[9] = self.rank;
        bytes[10] = self.world;
        bytes[12..14].copy_from_slice(&self.code.to_be_bytes());
        bytes[16..24].copy_from_slice(&self.local_cuda_device.to_be_bytes());
        bytes[24..26].copy_from_slice(&self.run_id_len.to_be_bytes());
        bytes[32..].copy_from_slice(&self.run_id);
        bytes
    }

    fn decode(bytes: [u8; STARTUP_BYTES]) -> Result<Self, ContextError> {
        if bytes[..8] != PROTOCOL_MAGIC {
            return Err(ContextError::Protocol("rendezvous magic mismatch"));
        }
        let run_id_len = u16::from_be_bytes([bytes[24], bytes[25]]);
        if usize::from(run_id_len) > MAX_RUN_ID_BYTES {
            return Err(ContextError::Protocol(
                "rendezvous run identity is too long",
            ));
        }
        let mut run_id = [0; MAX_RUN_ID_BYTES];
        run_id.copy_from_slice(&bytes[32..]);
        Ok(Self {
            kind: MessageKind::decode(bytes[8])?,
            rank: bytes[9],
            world: bytes[10],
            code: u16::from_be_bytes([bytes[12], bytes[13]]),
            local_cuda_device: u64::from_be_bytes(
                bytes[16..24]
                    .try_into()
                    .map_err(|_| ContextError::Protocol("invalid rendezvous device field"))?,
            ),
            run_id_len,
            run_id,
        })
    }

    fn run_id(&self) -> &[u8] {
        &self.run_id[..usize::from(self.run_id_len)]
    }
}

pub(super) fn validate_startup(
    message: &StartupMessage,
    run_id: &RunId,
    expected_rank: usize,
) -> Result<(), ContextError> {
    if message.world as usize != TWO_RANK_WORLD {
        return Err(ContextError::WorldSize {
            expected: TWO_RANK_WORLD,
            found: message.world as usize,
        });
    }
    if message.rank as usize != expected_rank {
        return Err(ContextError::RemoteRank {
            expected: expected_rank,
            found: message.rank as usize,
        });
    }
    if message.run_id() != run_id.as_str().as_bytes() {
        return Err(ContextError::RunIdMismatch);
    }
    if !matches!(message.kind, MessageKind::Hello | MessageKind::Accepted) {
        return Err(ContextError::Protocol(
            "unexpected message during rendezvous",
        ));
    }
    Ok(())
}

pub(super) fn write_startup(
    stream: &mut TcpStream,
    message: StartupMessage,
) -> Result<(), ContextError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| network("write rendezvous startup", error))
}

pub(super) fn read_startup(stream: &mut TcpStream) -> Result<StartupMessage, ContextError> {
    let mut bytes = [0; STARTUP_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| network("read rendezvous startup", error))?;
    StartupMessage::decode(bytes)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ControlMessage {
    pub(super) kind: MessageKind,
    pub(super) code: u16,
}

impl ControlMessage {
    pub(super) const fn shutdown() -> Self {
        Self {
            kind: MessageKind::Shutdown,
            code: 0,
        }
    }

    pub(super) const fn abort(failure: ContextFailure) -> Self {
        Self {
            kind: MessageKind::Abort,
            code: failure as u16,
        }
    }

    fn encode(self) -> [u8; CONTROL_BYTES] {
        let mut bytes = [0; CONTROL_BYTES];
        bytes[..8].copy_from_slice(&PROTOCOL_MAGIC);
        bytes[8] = self.kind as u8;
        bytes[10..12].copy_from_slice(&self.code.to_be_bytes());
        bytes
    }

    fn decode(bytes: [u8; CONTROL_BYTES]) -> Result<Self, ContextError> {
        if bytes[..8] != PROTOCOL_MAGIC {
            return Err(ContextError::Protocol("control message magic mismatch"));
        }
        Ok(Self {
            kind: MessageKind::decode(bytes[8])?,
            code: u16::from_be_bytes([bytes[10], bytes[11]]),
        })
    }
}

pub(super) fn write_control(
    stream: &mut TcpStream,
    message: ControlMessage,
) -> Result<(), ContextError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| network("write rendezvous control message", error))
}

pub(super) fn read_control(stream: &mut TcpStream) -> Result<ControlMessage, ContextError> {
    let mut bytes = [0; CONTROL_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| network("read rendezvous control message", error))?;
    ControlMessage::decode(bytes)
}
