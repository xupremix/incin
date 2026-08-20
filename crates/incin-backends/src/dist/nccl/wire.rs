//! Wire protocol structures and socket exchange helpers for NCCL bootstrap.

use core::ffi::c_char;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

use cudarc::nccl::Id;
use incin_core::dist::mesh::{DeviceIdentity, MeshId, TransportVersion};
use incin_core::dist::{AgreedPlan, PlanSummary, preflight};
use incin_core::tensor::device::DeviceId;

use crate::dist::nccl::config::{
    ARCHITECTURE_BYTES, BootstrapRole, LIBRARY_BYTES, MAGIC, PERSISTENT_BYTES, TOPOLOGY_MAGIC,
    TOPOLOGY_WIRE_BYTES, TwoRankBootstrapConfig, UNIQUE_ID_BYTES, WIRE_BYTES, WORLD,
};
use crate::dist::nccl::error::{NcclTransportError, io_error};
use crate::dist::nccl::topology::NcclTopology;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TopologyWire {
    pub(crate) rank: u8,
    pub(crate) world: u8,
    pub(crate) major: u32,
    pub(crate) minor: u32,
    pub(crate) patch: u32,
    pub(crate) persistent_len: u16,
    pub(crate) architecture_len: u16,
    pub(crate) library_len: u16,
    pub(crate) persistent: [u8; PERSISTENT_BYTES],
    pub(crate) architecture: [u8; ARCHITECTURE_BYTES],
    pub(crate) library: [u8; LIBRARY_BYTES],
}

impl TopologyWire {
    pub(crate) fn new(
        rank: u8,
        identity: &DeviceIdentity,
        transport: &TransportVersion,
    ) -> Result<Self, NcclTransportError> {
        let (persistent, persistent_len) =
            fixed_string::<PERSISTENT_BYTES>("persistent CUDA identity", identity.persistent())?;
        let (architecture, architecture_len) =
            fixed_string::<ARCHITECTURE_BYTES>("CUDA architecture", identity.architecture())?;
        let (library, library_len) =
            fixed_string::<LIBRARY_BYTES>("transport library", transport.library())?;
        let (major, minor, patch) = transport.version();
        Ok(Self {
            rank,
            world: WORLD as u8,
            major,
            minor,
            patch,
            persistent_len,
            architecture_len,
            library_len,
            persistent,
            architecture,
            library,
        })
    }

    pub(crate) fn identity(self) -> Result<DeviceIdentity, NcclTransportError> {
        Ok(DeviceIdentity::new(
            DeviceId::cuda(self.rank as usize),
            decode_fixed(
                &self.persistent,
                self.persistent_len,
                "persistent CUDA identity",
            )?,
            decode_fixed(
                &self.architecture,
                self.architecture_len,
                "CUDA architecture",
            )?,
        ))
    }

    pub(crate) fn transport(self) -> Result<TransportVersion, NcclTransportError> {
        Ok(TransportVersion::new(
            decode_fixed(&self.library, self.library_len, "transport library")?,
            self.major,
            self.minor,
            self.patch,
        ))
    }

    pub(crate) fn encode(self) -> [u8; TOPOLOGY_WIRE_BYTES] {
        let mut bytes = [0; TOPOLOGY_WIRE_BYTES];
        bytes[..8].copy_from_slice(&TOPOLOGY_MAGIC);
        bytes[8] = self.rank;
        bytes[9] = self.world;
        bytes[16..20].copy_from_slice(&self.major.to_be_bytes());
        bytes[20..24].copy_from_slice(&self.minor.to_be_bytes());
        bytes[24..28].copy_from_slice(&self.patch.to_be_bytes());
        bytes[28..30].copy_from_slice(&self.persistent_len.to_be_bytes());
        bytes[30..32].copy_from_slice(&self.architecture_len.to_be_bytes());
        bytes[32..34].copy_from_slice(&self.library_len.to_be_bytes());
        let persistent_end = 36 + PERSISTENT_BYTES;
        let architecture_end = persistent_end + ARCHITECTURE_BYTES;
        bytes[36..persistent_end].copy_from_slice(&self.persistent);
        bytes[persistent_end..architecture_end].copy_from_slice(&self.architecture);
        bytes[architecture_end..].copy_from_slice(&self.library);
        bytes
    }

    pub(crate) fn decode(bytes: [u8; TOPOLOGY_WIRE_BYTES]) -> Result<Self, NcclTransportError> {
        if bytes[..8] != TOPOLOGY_MAGIC {
            return Err(NcclTransportError::Protocol(
                "topology bootstrap magic mismatch",
            ));
        }
        let read_u32 = |start: usize| {
            let mut value = [0; 4];
            value.copy_from_slice(&bytes[start..start + 4]);
            u32::from_be_bytes(value)
        };
        let read_u16 = |start: usize| {
            let mut value = [0; 2];
            value.copy_from_slice(&bytes[start..start + 2]);
            u16::from_be_bytes(value)
        };
        let persistent_end = 36 + PERSISTENT_BYTES;
        let architecture_end = persistent_end + ARCHITECTURE_BYTES;
        let mut persistent = [0; PERSISTENT_BYTES];
        persistent.copy_from_slice(&bytes[36..persistent_end]);
        let mut architecture = [0; ARCHITECTURE_BYTES];
        architecture.copy_from_slice(&bytes[persistent_end..architecture_end]);
        let mut library = [0; LIBRARY_BYTES];
        library.copy_from_slice(&bytes[architecture_end..]);
        Ok(Self {
            rank: bytes[8],
            world: bytes[9],
            major: read_u32(16),
            minor: read_u32(20),
            patch: read_u32(24),
            persistent_len: read_u16(28),
            architecture_len: read_u16(30),
            library_len: read_u16(32),
            persistent,
            architecture,
            library,
        })
    }
}

pub(crate) fn fixed_string<const N: usize>(
    field: &'static str,
    value: &str,
) -> Result<([u8; N], u16), NcclTransportError> {
    if value.len() > N || value.len() > u16::MAX as usize {
        return Err(NcclTransportError::FieldTooLong {
            field,
            maximum: N,
            found: value.len(),
        });
    }
    let mut bytes = [0; N];
    bytes[..value.len()].copy_from_slice(value.as_bytes());
    Ok((bytes, value.len() as u16))
}

pub(crate) fn decode_fixed<const N: usize>(
    bytes: &[u8; N],
    len: u16,
    field: &'static str,
) -> Result<String, NcclTransportError> {
    let len = usize::from(len);
    if len > N {
        return Err(NcclTransportError::FieldTooLong {
            field,
            maximum: N,
            found: len,
        });
    }
    std::str::from_utf8(&bytes[..len])
        .map(str::to_owned)
        .map_err(|_| NcclTransportError::Protocol("topology field is not UTF-8"))
}

pub(crate) fn validate_topology_wire(
    message: &TopologyWire,
    expected_rank: u8,
) -> Result<(), NcclTransportError> {
    if message.world as usize != WORLD {
        return Err(NcclTransportError::WorldSize {
            expected: WORLD,
            found: message.world as usize,
        });
    }
    if message.rank != expected_rank {
        return Err(NcclTransportError::RemoteRank {
            expected: expected_rank as usize,
            found: message.rank as usize,
        });
    }
    Ok(())
}

pub(crate) fn write_topology_wire(
    stream: &mut TcpStream,
    message: TopologyWire,
) -> Result<(), NcclTransportError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| io_error("write topology bootstrap", error))
}

pub(crate) fn read_topology_wire(
    stream: &mut TcpStream,
) -> Result<TopologyWire, NcclTransportError> {
    let mut bytes = [0; TOPOLOGY_WIRE_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| io_error("read topology bootstrap", error))?;
    TopologyWire::decode(bytes)
}

pub(crate) fn format_transport(transport: &TransportVersion) -> String {
    let (major, minor, patch) = transport.version();
    format!("{} {major}.{minor}.{patch}", transport.library())
}

#[derive(Debug)]
pub(crate) struct BootstrapResult {
    pub(crate) unique_id: [u8; UNIQUE_ID_BYTES],
    pub(crate) agreed: AgreedPlan,
}

pub(crate) fn exchange_topology(
    config: TwoRankBootstrapConfig,
    local_identity: DeviceIdentity,
    local_transport: TransportVersion,
) -> Result<NcclTopology, NcclTransportError> {
    let local = TopologyWire::new(config.rank() as u8, &local_identity, &local_transport)?;
    let remote = match config.role {
        BootstrapRole::Root { bind } => {
            let listener = TcpListener::bind(bind).map_err(|error| io_error("bind", error))?;
            listener
                .set_nonblocking(true)
                .map_err(|error| io_error("set nonblocking", error))?;
            let mut stream = accept_until(&listener, config.timeout)?;
            drop(listener);
            configure_stream(&stream, config.timeout)?;
            let remote = read_topology_wire(&mut stream)?;
            validate_topology_wire(&remote, 1)?;
            write_topology_wire(&mut stream, local)?;
            remote
        }
        BootstrapRole::Peer { root } => {
            let mut stream = connect_until(root, config.timeout)?;
            configure_stream(&stream, config.timeout)?;
            write_topology_wire(&mut stream, local)?;
            let remote = read_topology_wire(&mut stream)?;
            validate_topology_wire(&remote, 0)?;
            remote
        }
    };
    let remote_transport = remote.transport()?;
    if remote_transport != local_transport {
        return Err(NcclTransportError::TransportMismatch {
            local: format_transport(&local_transport),
            remote: format_transport(&remote_transport),
        });
    }
    let remote_identity = remote.identity()?;
    let identities = if config.rank() == 0 {
        [local_identity, remote_identity]
    } else {
        [remote_identity, local_identity]
    };
    NcclTopology::new(identities, config.rank(), local_transport)
}

pub(crate) fn exchange_bootstrap(
    config: TwoRankBootstrapConfig,
    local: PlanSummary,
    root_id: Option<[u8; UNIQUE_ID_BYTES]>,
) -> Result<BootstrapResult, NcclTransportError> {
    if config.timeout.is_zero() {
        return Err(NcclTransportError::InvalidTimeout);
    }
    match config.role {
        BootstrapRole::Root { bind } => {
            let id = root_id.ok_or(NcclTransportError::MissingRootId)?;
            let listener = TcpListener::bind(bind).map_err(|error| io_error("bind", error))?;
            listener
                .set_nonblocking(true)
                .map_err(|error| io_error("set nonblocking", error))?;
            let mut stream = accept_until(&listener, config.timeout)?;
            drop(listener);
            configure_stream(&stream, config.timeout)?;
            let peer = read_wire(&mut stream)?;
            validate_wire(&peer, 1)?;
            let remote = peer.summary()?;
            let agreed = preflight(WORLD, &[local, remote])?;
            write_wire(&mut stream, WireMessage::new(0, local, id))?;
            Ok(BootstrapResult {
                unique_id: id,
                agreed,
            })
        }
        BootstrapRole::Peer { root } => {
            if root_id.is_some() {
                return Err(NcclTransportError::UnexpectedPeerId);
            }
            let mut stream = connect_until(root, config.timeout)?;
            configure_stream(&stream, config.timeout)?;
            write_wire(
                &mut stream,
                WireMessage::new(1, local, [0; UNIQUE_ID_BYTES]),
            )?;
            let root = read_wire(&mut stream)?;
            validate_wire(&root, 0)?;
            let remote = root.summary()?;
            let agreed = preflight(WORLD, &[remote, local])?;
            Ok(BootstrapResult {
                unique_id: root.unique_id,
                agreed,
            })
        }
    }
}

pub(crate) fn accept_until(
    listener: &TcpListener,
    timeout: Duration,
) -> Result<TcpStream, NcclTransportError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(NcclTransportError::InvalidTimeout)?;
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(NcclTransportError::Timeout {
                        phase: "accept rank one",
                        timeout,
                    });
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(io_error("accept", error)),
        }
    }
}

pub(crate) fn connect_until(
    root: SocketAddr,
    timeout: Duration,
) -> Result<TcpStream, NcclTransportError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(NcclTransportError::InvalidTimeout)?;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(NcclTransportError::Timeout {
                phase: "connect to rank zero",
                timeout,
            });
        }
        let remaining = deadline.saturating_duration_since(now);
        match TcpStream::connect_timeout(&root, remaining.min(Duration::from_millis(100))) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::ConnectionRefused
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::WouldBlock
                ) =>
            {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(io_error("connect", error)),
        }
    }
}

pub(crate) fn configure_stream(
    stream: &TcpStream,
    timeout: Duration,
) -> Result<(), NcclTransportError> {
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| io_error("set read timeout", error))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| io_error("set write timeout", error))?;
    stream
        .set_nodelay(true)
        .map_err(|error| io_error("set TCP_NODELAY", error))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WireMessage {
    pub(crate) rank: u8,
    pub(crate) world: u8,
    pub(crate) mesh: u64,
    pub(crate) hash: u64,
    pub(crate) collectives: u64,
    pub(crate) unique_id: [u8; UNIQUE_ID_BYTES],
}

impl WireMessage {
    pub(crate) fn new(rank: u8, summary: PlanSummary, unique_id: [u8; UNIQUE_ID_BYTES]) -> Self {
        Self {
            rank,
            world: WORLD as u8,
            mesh: summary.mesh_id().digest(),
            hash: summary.hash(),
            collectives: summary.collective_count() as u64,
            unique_id,
        }
    }

    pub(crate) fn summary(self) -> Result<PlanSummary, NcclTransportError> {
        let collectives = usize::try_from(self.collectives)
            .map_err(|_| NcclTransportError::Protocol("collective count exceeds usize"))?;
        Ok(PlanSummary::from_parts(
            MeshId::from_digest(self.mesh),
            self.hash,
            collectives,
        ))
    }

    pub(crate) fn encode(self) -> [u8; WIRE_BYTES] {
        let mut bytes = [0; WIRE_BYTES];
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8] = self.rank;
        bytes[9] = self.world;
        bytes[16..24].copy_from_slice(&self.mesh.to_be_bytes());
        bytes[24..32].copy_from_slice(&self.hash.to_be_bytes());
        bytes[32..40].copy_from_slice(&self.collectives.to_be_bytes());
        bytes[40..].copy_from_slice(&self.unique_id);
        bytes
    }

    pub(crate) fn decode(bytes: [u8; WIRE_BYTES]) -> Result<Self, NcclTransportError> {
        if bytes[..8] != MAGIC {
            return Err(NcclTransportError::Protocol("bootstrap magic mismatch"));
        }
        let read_u64 = |start: usize| {
            let mut value = [0; 8];
            value.copy_from_slice(&bytes[start..start + 8]);
            u64::from_be_bytes(value)
        };
        let mut unique_id = [0; UNIQUE_ID_BYTES];
        unique_id.copy_from_slice(&bytes[40..]);
        Ok(Self {
            rank: bytes[8],
            world: bytes[9],
            mesh: read_u64(16),
            hash: read_u64(24),
            collectives: read_u64(32),
            unique_id,
        })
    }
}

pub(crate) fn validate_wire(
    message: &WireMessage,
    expected_rank: u8,
) -> Result<(), NcclTransportError> {
    if message.world as usize != WORLD {
        return Err(NcclTransportError::WorldSize {
            expected: WORLD,
            found: message.world as usize,
        });
    }
    if message.rank != expected_rank {
        return Err(NcclTransportError::RemoteRank {
            expected: expected_rank as usize,
            found: message.rank as usize,
        });
    }
    Ok(())
}

pub(crate) fn write_wire(
    stream: &mut TcpStream,
    message: WireMessage,
) -> Result<(), NcclTransportError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| io_error("write bootstrap", error))
}

pub(crate) fn read_wire(stream: &mut TcpStream) -> Result<WireMessage, NcclTransportError> {
    let mut bytes = [0; WIRE_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| io_error("read bootstrap", error))?;
    WireMessage::decode(bytes)
}

pub(crate) fn id_to_bytes(id: &Id) -> [u8; UNIQUE_ID_BYTES] {
    let mut bytes = [0; UNIQUE_ID_BYTES];
    for (destination, source) in bytes.iter_mut().zip(id.internal()) {
        *destination = *source as u8;
    }
    bytes
}

pub(crate) fn id_from_bytes(bytes: [u8; UNIQUE_ID_BYTES]) -> Id {
    let mut internal = [0 as c_char; UNIQUE_ID_BYTES];
    for (destination, source) in internal.iter_mut().zip(bytes) {
        *destination = source as c_char;
    }
    Id::uninit(internal)
}

pub(crate) fn format_cuda_uuid(bytes: [c_char; 16]) -> String {
    use core::fmt::Write as _;

    let mut output = String::with_capacity(36);
    for (index, byte) in bytes.into_iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            output.push('-');
        }
        let _ = write!(output, "{:02x}", byte as u8);
    }
    output
}
