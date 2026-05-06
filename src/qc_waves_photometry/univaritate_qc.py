import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa


class ColumnQC:
    # handle NaNs. 
    def __init__ (self, column_name, file_path, index_mask = None, logged = False):
        self.column_name = column_name
        self.file_path = file_path
        self.index_mask = index_mask
        self.logged = logged


    def load_column(self):
        self.photom_col = pd.read_parquet(self.file_path, columns=[self.column_name])
        if self.index_mask is not None:
            self.photom_col = self.photom_col.loc[self.index_mask]
        return self.photom_col[self.column_name]

    
    def nan_fraction(self):
        total_count = len(self.photom_col[self.column_name])
        nan_count = self.photom_col[self.column_name].isna().sum()
        return nan_count / total_count if total_count > 0 else 0


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
        

    def stdev(self):
        return np.std(self.photom_col[self.column_name])
    

    def skewness(self):
        return stats.skew(self.photom_col[self.column_name])
    

    def kurtosis(self):
        return stats.kurtosis(self.photom_col[self.column_name])
    

    def mad(self):
        return np.median(np.abs(self.photom_col[self.column_name] - np.median(self.photom_col[self.column_name])))
    

    def iqr(self):
        q75, q25 = np.percentile(self.photom_col[self.column_name], [75 ,25])
        return q75 - q25
    

    def three_sigma_outliers(self):
        mean = np.mean(self.photom_col[self.column_name])
        std_dev = self.stdev()
        outliers = self.photom_col[np.abs(self.photom_col[self.column_name] - mean) > 3 * std_dev]
        return outliers
    

    def clean_up_memory(self):
        del self.photom_col
        del self.index_mask


class UnivariatePhotomQC:
    def __init__(self, region_file_path='/Users/sp624AA/Downloads/waves_qc/photometry_WD01.parquet'):
        self.region_file_path = region_file_path
        self.index_mask = None


    def get_column_names(self):
        return pq.read_schema(self.region_file_path).names
    

    def get_column_types(self):
        return pq.read_schema(self.region_file_path).types
    

    def get_column_lenth(self):
        return pq.read_table(self.region_file_path, columns=[self.get_column_names()[0]]).num_rows


    def sort_columns(self):
        bags_of_columns = {
            'sky_coordinates': None,
            'fluxes': None,
            'magnitudes': None,
            'seeings': None,
            'radii': None,
            'flags': None,
            'misc_floats': None,
            'misc_ints': None,
            'misc_strings': None
        }

        col_names = self.get_column_names()
        col_types = {name: t for name, t in zip(col_names, self.get_column_types())}
        remaining_cols = list(col_names)

        # Name-based theme assignment
        name_based = {
            'sky_coordinates': lambda col: 'ra_' in col or 'dec_' in col,
            'fluxes':          lambda col: 'flux_' in col,
            'magnitudes':      lambda col: 'mag_' in col,
            'seeings':         lambda col: 'seeing_' in col or 'sky_' in col,
            'radii':           lambda col: 'radius_' in col,
            'flags':           lambda col: 'flag_' in col or 'mask_' in col,
        }

        for bag, match_fn in name_based.items():
            bags_of_columns[bag] = [col for col in col_names if match_fn(col)]
            remaining_cols = [col for col in remaining_cols if not match_fn(col)]

        # Type-based assignment for remaining columns
        float_types = (pa.float32(), pa.float64())
        int_types   = (pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                       pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64())

        misc_floats, misc_ints, misc_strings = [], [], []
        for col in remaining_cols:
            t = col_types[col]
            if t in float_types:
                misc_floats.append(col)
            elif t in int_types:
                misc_ints.append(col)
            else:
                misc_strings.append(col)

        bags_of_columns['misc_floats']  = misc_floats
        bags_of_columns['misc_ints']    = misc_ints
        bags_of_columns['misc_strings'] = misc_strings

        # --- Checks ---
        all_assigned = [col for cols in bags_of_columns.values() for col in cols]

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

        return bags_of_columns

            
    def get_flagged_indexs(self, selection):
        possible_masks = ['mask', 'starmask', 'ghostmask', 'duplicate', 'patch', 'artefact']
        if selection not in possible_masks:
            raise ValueError(f"Selection must be one of {possible_masks}")
        
        length = self.get_column_lenth()
        selection = np.ones(length, dtype=bool)  # Start with all True
        for col_sel in selection:
            sel_name = f'flag_{col_sel}'
            column_selection = pd.read_parquet(self.region_file_path, columns = [sel_name])[sel_name] == 1
            selection &= column_selection.values  # Combine with AND

        return selection
    
        

        
