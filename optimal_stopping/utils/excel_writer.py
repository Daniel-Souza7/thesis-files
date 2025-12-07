"""Excel output utilities for experiment results"""

import pandas as pd
from datetime import datetime
import os


class ExcelWriter:
    """
    Utility class for writing experiment results to Excel
    """

    def __init__(self, output_dir='results'):
        """
        Parameters:
        -----------
        output_dir : str
            Directory to save Excel files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_results(self, results, filename=None):
        """
        Write results to Excel file

        Parameters:
        -----------
        results : list of dict
            List of experiment results, each dict containing:
            - method: name of the method
            - option_type: call/put
            - payoff_type: basket/geometric
            - n_assets: number of assets
            - N: number of time steps
            - price: option price
            - execution_time: time in seconds
            - other parameters...

        filename : str, optional
            Output filename. If None, generates timestamp-based name.
        """
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'option_pricing_results_{timestamp}.xlsx'

        filepath = os.path.join(self.output_dir, filename)

        # Write to Excel with formatting
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main results
            df.to_excel(writer, sheet_name='Results', index=False)

            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Results']

            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"Results written to {filepath}")
        return filepath

    def write_comparison(self, results, filename=None):
        """
        Write comparison results across different methods

        Parameters:
        -----------
        results : dict
            Dictionary mapping method names to their results
        filename : str, optional
            Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'method_comparison_{timestamp}.xlsx'

        filepath = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for method_name, method_results in results.items():
                df = pd.DataFrame(method_results)
                df.to_excel(writer, sheet_name=method_name[:31], index=False)  # Excel sheet name limit

                # Auto-adjust columns
                worksheet = writer.sheets[method_name[:31]]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"Comparison results written to {filepath}")
        return filepath
