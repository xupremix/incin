#![cfg(any(feature = "cpu", feature = "cuda"))]
#![allow(clippy::needless_range_loop)]

use incin_backends::iteration::tile_2d;

#[test]
fn test_tile_2d_coverage_square() {
    const ROWS: usize = 128;
    const COLS: usize = 128;
    let mut visited = vec![vec![0usize; COLS]; ROWS];

    tile_2d::<32, 32>(ROWS, COLS, |r0, r1, c0, c1| {
        for r in r0..r1 {
            for c in c0..c1 {
                visited[r][c] += 1;
            }
        }
    });

    for r in 0..ROWS {
        for c in 0..COLS {
            assert_eq!(visited[r][c], 1, "Element ({r}, {c}) visited != 1 time");
        }
    }
}

#[test]
fn test_tile_2d_coverage_rectangular_non_multiple() {
    const ROWS: usize = 100;
    const COLS: usize = 75;
    let mut visited = vec![vec![0usize; COLS]; ROWS];

    tile_2d::<16, 32>(ROWS, COLS, |r0, r1, c0, c1| {
        for r in r0..r1 {
            for c in c0..c1 {
                visited[r][c] += 1;
            }
        }
    });

    for r in 0..ROWS {
        for c in 0..COLS {
            assert_eq!(visited[r][c], 1, "Element ({r}, {c}) visited != 1 time");
        }
    }
}

#[test]
fn test_tile_2d_coverage_edge_cases() {
    // 0x0
    let mut count = 0;
    tile_2d::<16, 16>(0, 0, |_, _, _, _| count += 1);
    assert_eq!(count, 0);

    // 1x1
    let mut visited = vec![vec![0usize; 1]; 1];
    tile_2d::<8, 8>(1, 1, |r0, r1, c0, c1| {
        for r in r0..r1 {
            for c in c0..c1 {
                visited[r][c] += 1;
            }
        }
    });
    assert_eq!(visited[0][0], 1);
}
