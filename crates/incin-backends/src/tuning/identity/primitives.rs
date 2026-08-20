//! Version representation and the length-delimited digest accumulator every
//! fingerprint's `digest()` method builds on.

pub(super) const IDENTITY_SCHEMA: &[u8] = b"incin.tuning.identity.v1";

/// A three-component software version used for drivers, compilers, and
/// transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoftwareVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

impl SoftwareVersion {
    /// Creates a version triple.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `(major, minor, patch)`.
    #[must_use]
    pub const fn components(self) -> (u32, u32, u32) {
        (self.major, self.minor, self.patch)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Digest(u64);

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub(super) const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    pub(super) fn field(self, bytes: &[u8]) -> Self {
        self.number(bytes.len() as u64).bytes(bytes)
    }

    pub(super) fn text(self, value: &str) -> Self {
        self.field(value.as_bytes())
    }

    pub(super) fn number(self, value: u64) -> Self {
        self.bytes(&value.to_le_bytes())
    }

    pub(super) fn version(self, version: SoftwareVersion) -> Self {
        self.number(u64::from(version.major))
            .number(u64::from(version.minor))
            .number(u64::from(version.patch))
    }

    pub(super) const fn finish(self) -> u64 {
        self.0
    }
}
