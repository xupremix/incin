//! The TCP rendezvous protocol driver: timeout/environment validation, the
//! accept/connect loops, and the handshake that produces a live context.
//!
//! This module is gated at the declaration site in `mod.rs`, so its items do
//! not repeat the `std` feature attribute individually.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use core::marker::PhantomData;
use core::sync::atomic::AtomicU8;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::STATE_ACTIVE;
use super::TWO_RANK_WORLD;
use super::error::{ContextError, network};
use super::identity::{DistributedIdentity, RunId};
use super::lifecycle::DistributedContext;
use super::rendezvous::RendezvousEndpoint;
use super::state::DistributedContextHandle;
use super::wire::{MessageKind, StartupMessage, read_startup, validate_startup, write_startup};

pub(super) fn validate_timeout(timeout: Duration) -> Result<(), ContextError> {
    if timeout.is_zero() || timeout.as_millis() > u128::from(u64::MAX) {
        Err(ContextError::InvalidTimeout)
    } else {
        Ok(())
    }
}

pub(super) fn validate_dyn_launch(
    endpoint: RendezvousEndpoint,
    rank: usize,
    world: usize,
    timeout: Duration,
) -> Result<(), ContextError> {
    if world != TWO_RANK_WORLD {
        return Err(ContextError::WorldSize {
            expected: TWO_RANK_WORLD,
            found: world,
        });
    }
    if rank >= world {
        return Err(ContextError::RankOutOfRange { rank, world });
    }
    if endpoint.rank() != rank {
        return Err(ContextError::RoleRankMismatch {
            role_rank: endpoint.rank(),
            rank,
        });
    }
    validate_timeout(timeout)
}

pub(super) fn read_env(name: &'static str) -> Result<String, ContextError> {
    std::env::var(name).map_err(|_| ContextError::MissingEnvironment { name })
}

pub(super) fn parse_env<T>(name: &'static str) -> Result<T, ContextError>
where
    T: core::str::FromStr,
{
    let value = read_env(name)?;
    value
        .parse()
        .map_err(|_| ContextError::InvalidEnvironment { name, value })
}

pub(super) fn rendezvous<M, R>(
    run_id: RunId,
    endpoint: RendezvousEndpoint,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    timeout: Duration,
) -> Result<DistributedContext<M, R>, ContextError> {
    validate_dyn_launch(endpoint, rank, world, timeout)?;
    let local = StartupMessage::hello(&run_id, rank, local_cuda_device)?;
    let (stream, remote) = match endpoint {
        RendezvousEndpoint::Root { bind } => {
            let listener =
                TcpListener::bind(bind).map_err(|error| network("bind rendezvous", error))?;
            listener
                .set_nonblocking(true)
                .map_err(|error| network("configure rendezvous listener", error))?;
            let mut stream = accept_until(&listener, timeout)?;
            drop(listener);
            configure_stream(&stream, timeout)?;
            let remote = read_startup(&mut stream)?;
            if let Err(error) = validate_startup(&remote, &run_id, 1) {
                let _ = write_startup(&mut stream, StartupMessage::reject(rejection_code(&error)));
                return Err(error);
            }
            write_startup(
                &mut stream,
                StartupMessage::accepted(&run_id, rank, local_cuda_device)?,
            )?;
            (stream, remote)
        }
        RendezvousEndpoint::Peer { root } => {
            let mut stream = connect_until(root, timeout)?;
            configure_stream(&stream, timeout)?;
            write_startup(&mut stream, local)?;
            let remote = read_startup(&mut stream)?;
            if remote.kind == MessageKind::Reject {
                return Err(ContextError::PeerRejected { code: remote.code });
            }
            validate_startup(&remote, &run_id, 0)?;
            if remote.kind != MessageKind::Accepted {
                return Err(ContextError::Protocol(
                    "rank zero did not accept rendezvous",
                ));
            }
            (stream, remote)
        }
    };
    configure_stream(&stream, timeout)?;
    // Disable Nagle for the tiny shutdown/abort messages. Failure is reported:
    // a context is not considered active unless its control path is usable.
    stream
        .set_nodelay(true)
        .map_err(|error| network("configure rendezvous control socket", error))?;

    Ok(DistributedContext {
        identity: DistributedIdentity {
            run_id,
            rank,
            world,
            local_cuda_device,
            peer_cuda_device: remote.local_cuda_device as usize,
        },
        handle: DistributedContextHandle {
            state: Arc::new(AtomicU8::new(STATE_ACTIVE)),
        },
        control: Arc::new(Mutex::new(stream)),
        endpoint,
        timeout,
        marker: PhantomData,
    })
}

fn rejection_code(error: &ContextError) -> u16 {
    match error {
        ContextError::RunIdMismatch => 1,
        ContextError::RemoteRank { .. } => 2,
        ContextError::WorldSize { .. } => 3,
        _ => 255,
    }
}

fn configure_stream(stream: &TcpStream, timeout: Duration) -> Result<(), ContextError> {
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| network("set rendezvous read timeout", error))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| network("set rendezvous write timeout", error))
}

fn accept_until(listener: &TcpListener, timeout: Duration) -> Result<TcpStream, ContextError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(ContextError::InvalidTimeout)?;
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(ContextError::Network {
                        phase: "accept rank one",
                        message: "rendezvous deadline elapsed".to_string(),
                    });
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(network("accept rank one", error)),
        }
    }
}

fn connect_until(root: SocketAddr, timeout: Duration) -> Result<TcpStream, ContextError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(ContextError::InvalidTimeout)?;
    loop {
        match TcpStream::connect_timeout(&root, timeout.min(Duration::from_millis(50))) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::NotConnected
                ) =>
            {
                if Instant::now() >= deadline {
                    return Err(ContextError::Network {
                        phase: "connect to rank zero",
                        message: "rendezvous deadline elapsed".to_string(),
                    });
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(network("connect to rank zero", error)),
        }
    }
}
