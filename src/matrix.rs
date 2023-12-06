use std::ops::{Index, IndexMut};

/// Matrix ordering
/// The matrix is stored in contiguous memory vector.
/// Then, there are two way to store a matrix:
///     - according to row, row major ordering
///     - according to column, column major ordering
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ordering {
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
    order: Ordering,
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
                order,
            },
            Ordering::ColumnMajor => Self {
                stride_row: 1,
                stride_col: nb_rows,
                order,
            },
        }
    }

    /// Compute index of data in vector which store matrix data
    /// from row index and colunm index
    fn index(&self, row_id: usize, col_id: usize) -> usize {
        return row_id * self.stride_row + col_id * self.stride_col;
    }
}

/// Matrix
/// The data is stored in contiguous memory vector
/// according to ordering chosen
pub struct Matrix<T> {
    nb_rows: usize,
    nb_cols: usize,
    storage_order: StorageOrder,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Default,
{
    // Matrix constructor
    // nb_rows and nb_cols correspond respectively to number of rows and columns of matrix
    // order is matrix ordering
    pub fn new(nb_rows: usize, nb_cols: usize, order: Ordering) -> Self {
        let mut data: Vec<T> = Vec::new();
        data.resize_with(nb_rows * nb_cols, Default::default);

        return Self {
            nb_rows,
            nb_cols,
            storage_order: StorageOrder::new(nb_rows, nb_cols, order),
            data,
        };
    }

    /// Get number of rows
    pub fn nb_rows(&self) -> usize {
        return self.nb_rows;
    }

    /// Get number of columns
    pub fn nb_cols(&self) -> usize {
        return self.nb_cols;
    }

    /// Get matrix ordering
    pub fn order(&self) -> Ordering {
        return self.storage_order.order;
    }
}

/// Implementation of Index trait for Matrix
/// This allows to read the matrix element at (index of row, index of column) position
/// like this let element: f32 = matrix[(0, 2)];
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let id: usize = self.storage_order.index(index.0, index.1);
        return self.data.index(id);
    }
}

/// Implementation of IndexMut trait for Matrix
/// This allows to write an value in matrix at (index of row, index of column) position
/// like this matrix[(0, 2)] = 3.1415;
impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let id: usize = self.storage_order.index(index.0, index.1);
        return self.data.index_mut(id);
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

    #[test]
    fn test_matrix_new() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let matrix: Matrix<i32> = Matrix::new(nb_rows, nb_cols, Ordering::RowMajor);

        assert_eq!(matrix.nb_rows, nb_rows);
        assert_eq!(matrix.nb_cols, nb_cols);
        assert_eq!(matrix.data.len(), nb_rows * nb_cols);
    }

    #[test]
    fn test_matrix_dimensions_accessors() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let matrix: Matrix<i32> = Matrix::new(nb_rows, nb_cols, Ordering::RowMajor);

        assert_eq!(matrix.nb_rows(), nb_rows);
        assert_eq!(matrix.nb_cols(), nb_cols);
    }

    #[test]
    fn test_matrix_order_accessor() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let order: Ordering = Ordering::ColumnMajor;
        let matrix: Matrix<i32> = Matrix::new(nb_rows, nb_cols, order);

        assert_eq!(matrix.order(), order);
    }

    #[test]
    fn test_matrix_index_accessors() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new(nb_rows, nb_cols, Ordering::RowMajor);

        let value: i32 = 1;

        matrix[(0, 0)] = value;
        assert_eq!(matrix[(0, 0)], value);

        matrix[(2, 1)] = value;
        assert_eq!(matrix[(2, 1)], value);

        matrix[(1, 2)] = value;
        assert_eq!(matrix[(1, 2)], value);
    }
}
