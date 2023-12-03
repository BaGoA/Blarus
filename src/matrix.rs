/// Matrix ordering
/// The matrix is stored in contiguous memory vector.
/// Then, there are two way to store a matrix:
///     - according to row, row major ordering
///     - according to column, column major ordering
enum Ordering {
    RowMajor,
    ColumnMajor,
}

/// Matrix storage order
/// These structure contain strides along row and column
/// that we need to apply to matrix indexes (i, j) to obtain the
/// index of data in vector which store matrix data
struct StorageOrder {
    stride_row: usize,
    stride_col: usize,
}

impl StorageOrder {
    // Matrix storage order constructor
    // nb_rows and nb_cols correspond respectively to number of rows and columns of matrix
    // order is matrix ordering
    fn new(nb_rows: usize, nb_cols: usize, order: Ordering) -> Self {
        match order {
            Ordering::RowMajor => Self {
                stride_row: nb_cols,
                stride_col: 1,
            },
            Ordering::ColumnMajor => Self {
                stride_row: 1,
                stride_col: nb_rows,
            },
        }
    }

    /// Compute index of data in vector which store matrix data
    /// from row index anf colunm index
    fn index(&self, row_id: usize, col_id: usize) -> usize {
        return row_id * self.stride_row + col_id * self.stride_col;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_order_new() {
        let nb_rows: usize = 2;
        let nb_cols: usize = 3;

        let mut storage_order = StorageOrder::new(nb_rows, nb_cols, Ordering::RowMajor);
        assert_eq!(storage_order.stride_row, nb_cols);
        assert_eq!(storage_order.stride_col, 1);

        storage_order = StorageOrder::new(nb_rows, nb_cols, Ordering::ColumnMajor);
        assert_eq!(storage_order.stride_row, 1);
        assert_eq!(storage_order.stride_col, nb_rows);
    }

    #[test]
    fn test_storage_order_index() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut storage_order = StorageOrder::new(nb_rows, nb_cols, Ordering::RowMajor);
        assert_eq!(storage_order.index(1, 2), nb_cols + 2);

        storage_order = StorageOrder::new(nb_rows, nb_cols, Ordering::ColumnMajor);
        assert_eq!(storage_order.index(2, 1), 2 + nb_rows);
    }
}
