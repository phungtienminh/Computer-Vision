class Solver:
    @staticmethod
    def solve(board):
        row_mask = [0 for _ in range(9)]
        col_mask = [0 for _ in range(9)]
        region_mask = [[0 for y in range(3)] for x in range(3)]

        for row in range(9):
            for col in range(9):
                if board[row][col] != 0:
                    row_mask[row] |= (1 << board[row][col])
                    col_mask[col] |= (1 << board[row][col])
                    region_mask[row // 3][col // 3] |= (1 << board[row][col])

        def go(row, col):
            if board[row][col] != 0:
                if row == 8 and col == 8:
                    return board
                if col == 8:
                    return go(row + 1, 0)
                else:
                    return go(row, col + 1)
            else:
                for digit in range(1, 10):
                    if (row_mask[row] & (1 << digit)) > 0 or \
                       (col_mask[col] & (1 << digit)) > 0 or \
                       (region_mask[row // 3][col // 3] & (1 << digit)) > 0:
                        continue

                    board[row][col] = digit
                    if row == 8 and col == 8:
                        return board

                    row_mask[row] |= (1 << digit)
                    col_mask[col] |= (1 << digit)
                    region_mask[row // 3][col // 3] |= (1 << digit)

                    result = go(row + 1, 0) if col == 8 else go(row, col + 1)
                    if result is not None:
                        return result

                    row_mask[row] &= ~(1 << digit)
                    col_mask[col] &= ~(1 << digit)
                    region_mask[row // 3][col // 3] &= ~(1 << digit)
                    board[row][col] = 0

            return None

        return go(0, 0)
