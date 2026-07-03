use kindle_core::prelude::*;
use rayon::prelude::*;

/// A strict-typed batch containing features and optional targets.
pub struct Batch<S1: Shape, S2: Shape = ()> {
    pub features: Tensor<S1>,
    pub targets: Option<Tensor<S2>>,
}

/// Extension trait to convert any standard Rust `Iterator` into a multi-threaded Parallel Dataloader.
pub trait DataLoaderExt: Iterator + Send + Sized {
    /// Converts this iterator into a parallel dataloader using `rayon`.
    /// 
    /// This effortlessly spreads the data loading across all available CPU cores.
    fn into_par_loader(self) -> rayon::iter::IterBridge<Self>
    where
        Self::Item: Send + Sync,
    {
        self.par_bridge()
    }
}

// Implement for all standard iterators automatically!
impl<I: Iterator + Send> DataLoaderExt for I {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_loader() {
        let items: Vec<i32> = (0..100).collect();
        let iter = items.into_iter();

        // Convert the iterator into a rayon Parallel Iterator natively
        let sum: i32 = iter.into_par_loader()
            .map(|x| x * 2)
            .sum();

        assert_eq!(sum, 9900);
    }
}
