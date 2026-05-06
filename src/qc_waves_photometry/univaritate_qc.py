import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
import os
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import argparse
import yaml


class ColumnQC:
    # This helper class handles QC operations for a *single column* at a time.
    # It deliberately encapsulates reading, filtering, stats, and plotting logic
    # so higher-level orchestration code can stay focused on "which columns to run".
    def __init__ (self, column_name, file_path, index_mask = None, logged = False):
        self.column_name = column_name
        self.file_path = file_path
        self.index_mask = index_mask
        self.logged = logged


    def load_column(self):
        # Read only the requested column from parquet to keep memory usage smaller.
        self.photom_col = pd.read_parquet(self.file_path, columns=[self.column_name])
        if self.index_mask is not None:
            # If a boolean mask is provided, keep only rows that pass the mask.
            self.photom_col = self.photom_col.loc[self.index_mask]

        if self.logged:
            # Optional log10 scaling for quantities such as fluxes/radii.
            # NOTE: this expects positive values; non-positive values would become
            # invalid/-inf and should be handled upstream if present.
            self.photom_col[self.column_name] = np.log10(self.photom_col[self.column_name])
        return self.photom_col[self.column_name]

    
    def nan_fraction(self):
        # Count total rows and rows with missing values, then convert to a fraction.
        # Returning 0 for empty input avoids divide-by-zero errors.
        total_count = len(self.photom_col[self.column_name])
        nan_count = self.photom_col[self.column_name].isna().sum()
        return nan_count / total_count if total_count > 0 else 0
    

    def nan_indexs(self):
        # Return the DataFrame index values for quick traceability back to source rows.
        return self.photom_col[self.column_name].isna().index.tolist()
    

    def drop_nans(self):
        # Most statistics are computed on finite data only, so we remove NaNs here.
        self.photom_col = self.photom_col.dropna(subset=[self.column_name])
        return self.photom_col[self.column_name]


    def plot_hist(self, save_location=None, density=False, log_scale=False):
        plt.clf()
        plt.hist(self.photom_col[self.column_name], bins=50, density=density)
        plt.xlabel(self.column_name)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {self.column_name}')
        if log_scale:
            plt.yscale('log')
        if save_location:
            plt.savefig(save_location)
        else:
            plt.show()
    

    def min(self):
        # Small helper wrappers keep all metric calculations in one class.
        return np.min(self.photom_col[self.column_name])
    

    def max(self):
        return np.max(self.photom_col[self.column_name])
    

    def mean(self):
        return np.mean(self.photom_col[self.column_name])
    

    def median(self):
        return np.median(self.photom_col[self.column_name])
    

    def stdev(self):
        return np.std(self.photom_col[self.column_name])
    

    def mad(self):
        return np.median(np.abs(self.photom_col[self.column_name] - np.median(self.photom_col[self.column_name])))
    

    def sigma_percentiles(self):
        return np.percentile(self.photom_col[self.column_name], [0, 16, 50, 84, 100])
    

    def quantiles(self):
        return np.percentile(self.photom_col[self.column_name], [0, 25, 50, 75 ,100])
    

    def percentiles(self):
        return np.percentile(self.photom_col[self.column_name], np.arange(0, 101, 1))


    def three_sigma_outliers(self):
        mean = self.mean()
        std_dev = self.stdev()
        outliers = len(self.photom_col[np.abs(self.photom_col[self.column_name] - mean) > 3 * std_dev]) / len(self.photom_col[self.column_name])
        return outliers
    

    def clean_up_memory(self):
        del self.photom_col
        del self.index_mask


class UnivariatePhotomQC:
    # Main orchestration class for bagging columns and generating all QC outputs.
    def __init__(self, region_file_path='/Users/sp624AA/Downloads/waves_qc/photometry_WD01.parquet',
                 region_maml_file_path='/Users/sp624AA/Downloads/waves_qc/photometry_WD01.maml',
                 region_name='WD01',
                 save_dir='/Users/sp624AA/Downloads/waves_qc/plots'):
        self.region_file_path = region_file_path
        self.region_maml_file_path = region_maml_file_path
        self.region_name = region_name
        self.save_dir = save_dir

        self.flux_masks = ['mask', 'starmask', 'artefact']
        self.mag_masks = ['mask', 'starmask', 'artefact']
        self.radii_masks = ['mask', 'starmask', 'artefact']

        self.coord_plots = {'pdf': None, 'bar': ['min', 'max', 'nan_fraction']}
        self.flux_plots = {'pdf': 'bag', 'bar': ['min', 'max', 'mean', 'median', 'stdev', 'mad', '3_sigma_outliers', 'nan_fraction']}
        self.mag_plots = {'pdf': 'bag', 'bar': ['min', 'max', 'mean', 'median', 'stdev', 'mad', '3_sigma_outliers', 'nan_fraction']}
        self.seeing_plots = {'pdf': 'bag', 'bar': ['min', 'max', 'mean', 'median', 'stdev', 'mad', '3_sigma_outliers', 'nan_fraction']}
        self.radii_plots = {'pdf': 'bag', 'bar': ['min', 'max', 'mean', 'median', 'stdev', 'mad', '3_sigma_outliers', 'nan_fraction']}
        self.flags_plots = {'bar': ['nan_fraction']}
        self.misc_floats_plots = {'pdf': 'single', 'bar': ['3_sigma_outliers', 'nan_fraction']}
        self.misc_ints_plots = {'pdf': None, 'bar': ['3_sigma_outliers', 'nan_fraction']}
        self.misc_strings_plots = {'pdf': None, 'bar': ['nan_fraction']}

        self.bags_of_columns = {
            # Each bag contains:
            # - columns: populated later by _sort_columns()
            # - logged: whether values are log10 transformed before computing stats/plots
            # - apply_flags: which flag columns must be true (==1) for row selection
            # - plots: which plot modes/metrics should be generated
            'sky_coordinates': {'columns': None, 'logged': False, 'apply_flags': None, 'plots': self.coord_plots},

            'total_fluxes': {'columns': None, 'logged': True, 'apply_flags': self.flux_masks, 'plots': self.flux_plots},

            'total_flux_errors': {'columns': None, 'logged': True, 'apply_flags': self.flux_masks, 'plots': self.flux_plots},

            'total_uncorrected_fluxes': {'columns': None, 'logged': True, 'apply_flags': self.flux_masks, 'plots': self.flux_plots},

            'total_uncorrected_flux_errors': {'columns': None, 'logged': True, 'apply_flags': self.flux_masks, 'plots': self.flux_plots},

            'colour_fluxes': {'columns': None, 'logged': True, 'apply_flags': self.flux_masks, 'plots': self.flux_plots},

            'colour_flux_errors': {'columns': None, 'logged': True, 'apply_flags': self.flux_masks, 'plots': self.flux_plots},

            'fibre_magnitudes': {'columns': None, 'logged': False, 'apply_flags': self.mag_masks, 'plots': self.mag_plots},

            'fibre_magnitude_errors': {'columns': None, 'logged': False, 'apply_flags': self.mag_masks, 'plots': self.mag_plots},

            'Z_magnitudes': {'columns': None, 'logged': False, 'apply_flags': self.mag_masks, 'plots': self.mag_plots},

            'seeings': {'columns': None, 'logged': False, 'apply_flags': None, 'plots': self.seeing_plots},

            'radii': {'columns': None, 'logged': True, 'apply_flags': self.radii_masks, 'plots': self.radii_plots},

            'flags': {'columns': None, 'logged': False, 'apply_flags': None, 'plots': self.flags_plots},

            'misc_floats': {'columns': None, 'logged': False, 'apply_flags': None, 'plots': self.misc_floats_plots},

            'misc_ints': {'columns': None, 'logged': False, 'apply_flags': None, 'plots': self.misc_ints_plots},

            'misc_strings': {'columns': None, 'logged': False, 'apply_flags': None, 'plots': self.misc_strings_plots}
        }

        # Build internal structures immediately during initialization so the object
        # is ready to run plots right away.
        self._sort_columns()
        self._get_unit_lookup_table()

    def get_column_names(self):
        # Read schema only (fast) to discover available column names.
        return pq.read_schema(self.region_file_path).names
    

    def get_column_types(self):
        # Read schema types in parallel with column names for downstream bucketing.
        return pq.read_schema(self.region_file_path).types
    

    def get_column_length(self):
        # Read only one column to get table row count without loading full dataset.
        return pq.read_table(self.region_file_path, columns=[self.get_column_names()[0]]).num_rows
    

    def _get_unit_lookup_table(self):
        # Parse the metadata file and build a dict from column name -> unit.
        # This is used to label axes in plots.
        with open(self.region_maml_file_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        maml_fields = metadata.get("fields", [])

        self.units_by_column = {
        field["name"]: field.get("unit")
        for field in maml_fields
        if "name" in field
    }

    def get_column_unit(self, column_name):
        # Unit may be missing in metadata; .get() safely returns None in that case.
        return self.units_by_column.get(column_name)


    def _sort_columns(self):
        # Group columns into themed "bags" based on naming conventions first,
        # then fall back to datatype-based buckets for anything left over.

        col_names = self.get_column_names()
        col_types = {name: t for name, t in zip(col_names, self.get_column_types())}
        remaining_cols = list(col_names)

        # Name-based theme assignment
        name_based = {
            'sky_coordinates':               lambda col: 'ra_' in col or 'dec_' in col,
            'total_fluxes':                  lambda col: 'flux_' in col and '_total' in col and '_err' not in col and '_uncorrected' not in col and '_colour' not in col,
            'total_flux_errors':             lambda col: 'flux_' in col and '_total' in col and '_err' in col and '_uncorrected' not in col and '_colour' not in col,
            'total_uncorrected_fluxes':      lambda col: 'flux_' in col and '_total' in col and '_uncorrected' in col and '_err' not in col,
            'total_uncorrected_flux_errors': lambda col: 'flux_' in col and '_total' in col and '_uncorrected' in col and '_err' in col,
            'colour_fluxes':                 lambda col: 'flux_' in col and '_colour' in col and '_err' not in col,
            'colour_flux_errors':            lambda col: 'flux_' in col and '_colour' in col and '_err' in col,
            'fibre_magnitudes':              lambda col: 'mag_fibre_' in col and '_err' not in col,
            'fibre_magnitude_errors':        lambda col: 'mag_fibre_' in col and '_err' in col,
            'Z_magnitudes':                  lambda col: 'mag_Z_' in col,
            'seeings':                       lambda col: 'seeing_' in col or 'sky_' in col,
            'radii':                         lambda col: 'radius_' in col,
            'flags':                         lambda col: 'flag_' in col or 'mask_' in col,
        }

        for bag, match_fn in name_based.items():
            # For each bag, collect matching columns and remove them from the
            # set of columns that still need assignment.
            matched = [col for col in col_names if match_fn(col)]
            self.bags_of_columns[bag]['columns'] = matched  # ← update 'columns' key
            remaining_cols = [col for col in remaining_cols if not match_fn(col)]

        # Type-based assignment for remaining columns
        float_types = (pa.float32(), pa.float64())
        int_types   = (pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                    pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64())

        misc_floats, misc_ints, misc_strings = [], [], []
        for col in remaining_cols:
            # Anything not caught by naming rules is categorized by parquet type.
            t = col_types[col]
            if t in float_types:
                misc_floats.append(col)
            elif t in int_types:
                misc_ints.append(col)
            else:
                misc_strings.append(col)

        self.bags_of_columns['misc_floats']['columns']  = misc_floats   # ← update 'columns' key
        self.bags_of_columns['misc_ints']['columns']    = misc_ints
        self.bags_of_columns['misc_strings']['columns'] = misc_strings

        # --- Sanity checks ---
        # These checks protect against silent misconfiguration.
        all_assigned = [col for bag in self.bags_of_columns.values() for col in bag['columns']]

        # 1. No column left unassigned
        unassigned = [col for col in col_names if col not in all_assigned]
        if unassigned:
            raise ValueError(f"The following columns were not assigned to any theme: {unassigned}")

        # 2. No column assigned to more than one theme
        seen, duplicates = set(), set()
        for col in all_assigned:
            if col in seen:
                duplicates.add(col)
            seen.add(col)
        if duplicates:
            raise ValueError(f"The following columns were assigned to multiple themes: {duplicates}")

        return self.bags_of_columns

            
    def get_flagged_indexs(self, selection_columns):
        # check selection columns is an array
        if not isinstance(selection_columns, (list, np.ndarray)):
            raise ValueError("Selection columns must be a list or numpy array")
        
        possible_masks = ['mask', 'starmask', 'ghostmask', 'duplicate', 'patch', 'artefact']
        for col in selection_columns:
            if col not in possible_masks:
                raise ValueError(f"Selection must be one of {possible_masks}")
        
        # Build one combined boolean mask (logical AND across selected flags).
        length = self.get_column_length()
        selection_indexs = np.ones(length, dtype=bool)  # Start with all True
        for col_sel in selection_columns:
            # Flag columns are named like "flag_mask", "flag_starmask", etc.
            sel_name = f'flag_{col_sel}'
            column_selection = pd.read_parquet(self.region_file_path, columns = [sel_name])[sel_name] == 1
            selection_indexs &= column_selection.values  # Combine with AND

        return selection_indexs


    def plot_pdfs_per_bag(self, bag_name, save_location=None):
        if bag_name not in self.bags_of_columns:
            raise ValueError(f"Bag name '{bag_name}' not found. Available bags: {list(self.bags_of_columns.keys())}")
        
        columns = self.bags_of_columns[bag_name]['columns']
        logged = self.bags_of_columns[bag_name]['logged']
        mask = self.bags_of_columns[bag_name]['apply_flags']
        # Build row-selection mask once per bag, then reuse for each column.
        if mask:
            index_mask = self.get_flagged_indexs(mask)
        else:
            index_mask = None

        if not columns:
            raise ValueError(f"No columns found in bag '{bag_name}'")

        # We summarize each column by percentiles and approximate shape using KDE.
        percentiles_dict = {}
        minmax_dict = {}
        for col in columns:
            # For each column:
            # 1) load data (+ optional row mask / log transform),
            # 2) remove NaNs,
            # 3) cache percentiles and min/max for plotting.
            col_qc = ColumnQC(column_name=col, file_path=self.region_file_path, index_mask=index_mask, logged=logged)
            col_qc.load_column()
            col_qc.drop_nans()
            percentiles_dict[col] = col_qc.percentiles()       # 100 percentiles for PDF shape
            minmax_dict[col] = col_qc.min(), col_qc.max()    # [min, max]
            col_qc.clean_up_memory()

        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.5), 6))
        positions = range(1, len(columns) + 1)

        for pos, col in zip(positions, columns):
            all_percentiles = percentiles_dict[col]   # shape (100,)
            p0, p100 = minmax_dict[col]

            # --- PDF-like shape via KDE on percentile samples ---
            # We estimate a smooth density profile from percentile samples, then
            # draw it as a one-sided violin for each column position.
            kde = gaussian_kde(all_percentiles)
            y_range = np.linspace(p0, p100, 101)
            density = kde(y_range)
            violin_width = 0.95
            density_scaled = density / density.max() * violin_width

            # Fill pdf
            pos = pos - 0.95/2
            ax.fill_betweenx(y_range, pos, pos + density_scaled,
                            alpha=0.4, color='black', zorder=2)
            ax.plot(pos + density_scaled, y_range, color='black', linewidth=0.8, zorder=2)

        units = self.get_column_unit(columns[0])  # Assuming all columns in the bag have the same unit
        # Configure all axis labels/ticks once after drawing every column.
        ax.set_xticks(list(positions))
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.set_xlim(0.5, len(columns) + 0.5)
        ax.set_xlabel('Columns')
        if logged:
            ax.set_ylabel(f'Log10([{units}])')
        else:
            ax.set_ylabel(f'[{units}]')
        ax.set_title(f'{self.region_name} - PDFs for {bag_name}\nMasked on: {mask}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_location:
            plt.savefig(save_location)
        else:
            plt.show()
        plt.close(fig)
        

    def plot_single_pdfs_per_bag(self, bag_name, save_location=None):
        if bag_name not in self.bags_of_columns:
            raise ValueError(f"Bag name '{bag_name}' not found. Available bags: {list(self.bags_of_columns.keys())}")

        columns = self.bags_of_columns[bag_name]['columns']
        logged = self.bags_of_columns[bag_name]['logged']
        mask = self.bags_of_columns[bag_name]['apply_flags']
        index_mask = self.get_flagged_indexs(mask) if mask else None

        if not columns:
            raise ValueError(f"No columns found in bag '{bag_name}'")


        for col in columns:
            # Generate one standalone PDF-style plot per column.
            col_qc = ColumnQC(column_name=col, file_path=self.region_file_path, index_mask=index_mask, logged=logged)
            values = col_qc.load_column().dropna().to_numpy()

            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot a normalized histogram and overlay KDE when enough variation exists.
            if values.size:
                # Histogram gives empirical distribution; KDE gives a smooth trend line.
                ax.hist(values, bins=50, density=True, color='black', alpha=0.35)
                if values.size > 1 and values.min() != values.max():
                    x_range = np.linspace(values.min(), values.max(), 200)
                    ax.plot(x_range, gaussian_kde(values)(x_range), color='black', linewidth=1.2)

            units = self.get_column_unit(col)
            if logged:
                ax.set_xlabel(f'Log10([{units}])')
            else:
                ax.set_xlabel(f'[{units}]')
            ax.set_ylabel('Density')
            ax.set_title(f'{self.region_name} - PDF for {col}\nGroup: {bag_name} | Masked on: {mask}')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()

            if save_location:
                col_save_location = save_location + f'_{col}.png'
                plt.savefig(col_save_location)
            else:
                plt.show()

            plt.close(fig)
            col_qc.clean_up_memory()


    def plot_bar_charts_per_bag(self, bag_name, attribute, save_location=None):

        if bag_name not in self.bags_of_columns:
            raise ValueError(f"Bag name '{bag_name}' not found. Available bags: {list(self.bags_of_columns.keys())}")
        
        columns = self.bags_of_columns[bag_name]['columns']
        logged = self.bags_of_columns[bag_name]['logged']
        mask = self.bags_of_columns[bag_name]['apply_flags']
        if mask:
            index_mask = self.get_flagged_indexs(mask)
        else:
            index_mask = None
        if not columns:
            raise ValueError(f"No columns found in bag '{bag_name}'")
        
        # Dispatch table mapping metric names to calculation callables.
        attribute_functions = {
            'min': lambda qc: qc.min(),
            'max': lambda qc: qc.max(),
            'mean': lambda qc: qc.mean(),
            'median': lambda qc: qc.median(),
            'stdev': lambda qc: qc.stdev(),
            'mad': lambda qc: qc.mad(),
            '3_sigma_outliers': lambda qc: qc.three_sigma_outliers(),
            'nan_fraction': lambda qc: qc.nan_fraction()
        }
        if attribute not in attribute_functions:
            raise ValueError(f"Attribute '{attribute}' not recognized. Available attributes: {list(attribute_functions.keys())}")
        attribute_values = []
        for col in columns:
            # Compute the selected metric independently for each column.
            col_qc = ColumnQC(column_name=col, file_path=self.region_file_path, index_mask=index_mask, logged=logged)
            col_qc.load_column()
            if attribute != 'nan_fraction':
                # NaN fraction must include NaNs; all other metrics are finite-only.
                col_qc.drop_nans()
            attribute_value = attribute_functions[attribute](col_qc)
            attribute_values.append(attribute_value)
            col_qc.clean_up_memory()
        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.5), 6))
        ax.bar(columns, attribute_values, color='black', alpha=0.7)
        units = self.get_column_unit(columns[0])  # Assuming all columns in the bag have the same unit
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.set_xlabel('Columns')
        if attribute == 'nan_fraction':
            ax.set_ylabel('Fraction')
        elif attribute == '3_sigma_outliers':
            ax.set_ylabel('Fraction')
        else:
            if logged:
                ax.set_ylabel(f'Log10([{units}])')
            else:
                ax.set_ylabel(f'[{units}]')

        ax.set_title(f'{self.region_name} - {attribute} for {bag_name}\nMasked on: {mask}')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_location:
            plt.savefig(save_location)
        else:
            plt.show()
        plt.close(fig)

    def make_all_plots(self):
        # Iterate over all bags and emit each configured plot type.
        for bag_name, bag_info in self.bags_of_columns.items():
            plots = bag_info['plots']
            if 'pdf' in plots and plots['pdf'] == 'bag':
                # One combined PDF summary image for the entire bag.
                save_loc = os.path.join(self.save_dir, f'{bag_name}/pdfs/{self.region_name}/{bag_name}_pdfs.png')
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_loc), exist_ok=True)
                self.plot_pdfs_per_bag(bag_name, save_location=save_loc)


            elif 'pdf' in plots and plots['pdf'] == 'single':
                # One file per column for bags configured as "single".
                save_loc = os.path.join(self.save_dir, f'{bag_name}/pdfs/{self.region_name}/single_pdf')
                # create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_loc), exist_ok=True)
                self.plot_single_pdfs_per_bag(bag_name, save_location=save_loc)
            
            if 'bar' in plots:
                # Emit each requested bar-chart metric for this bag.
                for attribute in plots['bar']:
                    self.save_loc = os.path.join(self.save_dir, f'{bag_name}/bar_charts/{self.region_name}/{bag_name}_{attribute}_bar.png')
                    # create directory if it doesn't exist
                    os.makedirs(os.path.dirname(self.save_loc), exist_ok=True)
                    self.plot_bar_charts_per_bag(bag_name, attribute, save_location=self.save_loc)


def main():
    # CLI entry point: parse arguments, build QC object, run all configured plots.
    argparser = argparse.ArgumentParser(description='Univariate QC for photometry data')
    argparser.add_argument('--region_file_path', type=str, default='/Users/sp624AA/Downloads/waves_qc/photometry_WD01.parquet', help='Path to the region parquet file')
    argparser.add_argument('--region_maml_file_path', type=str, default='/Users/sp624AA/Downloads/waves_qc/photometry_WD01.maml', help='Path to the region maml file')
    argparser.add_argument('--region_name', type=str, default='WD01', help='Name of the region')
    argparser.add_argument('--save_dir', type=str, default='/Users/sp624AA/Downloads/waves_qc/plots', help='Directory to save the plots')
    args = argparser.parse_args()

    # Run full orchestration: initialize object (which sorts columns + units),
    # then generate all requested outputs.
    print(f"Running univariate QC for region: {args.region_name}")
    qc = UnivariatePhotomQC(region_file_path=args.region_file_path, region_maml_file_path=args.region_maml_file_path, region_name=args.region_name)
    qc.make_all_plots()
    print('Done!')

if __name__ == "__main__":
    main()
