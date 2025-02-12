import os
import ast
import json
import traceback
import yaml
import itertools
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    ''' Example usage:
    python3 ~/git/scclv2/src/cl_replay/api/utils/eval_exp.py --results_dir /path/to/dir/ --save_path /path/to/dir/ --custom_str _test
    '''
    parser = ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str, help="rel. path to the parent directory holding exp. results.")
    parser.add_argument("--save_path", required=True, type=str, help="rel. save path of global result file.")
    parser.add_argument("--custom_str", type=str, default="", help="if any additional custom strings are attached to the .csv file descriptors.")
    parser.add_argument("--select_groups", nargs='*', required=False, type=int, help="specify which groups should be evaluated")
    parser.add_argument("--select_yaml_fields", nargs='*', required=False, type=str, help="specify args to list in yaml for each combinatorial run.")
    parser.add_argument("--filter_metrics", nargs ='*', default=['acc'], type=str, help='pass metrics to filter on aggregation.')
    parser_args = parser.parse_args()

    db_file = pd.read_csv(os.path.join(parser_args.results_dir, 'db.csv'))
    uniq_grps = db_file['group_id'].unique() # get unique experiment groups via "exp_group_id" field
    selected_groups = []
    if parser_args.select_groups:
        for gid in uniq_grps:
            if gid in parser_args.select_groups:
                selected_groups.append(gid)
    else: selected_groups = uniq_grps
    if not os.path.isdir(parser_args.save_path): os.makedirs(parser_args.save_path, exist_ok=True)
    # ------------------------------------------ PARAMS & COLUMNS
    # TODO: Current approach does not work to evaluate different task-splits at once
    # * since column headers have different lengths
    # * solution requires to pad the data at the correct place
    col_data = []; global_param_set = {}
    for grp_id in selected_groups:
        combs = db_file.loc[(db_file['group_id'] == grp_id)]
        uniq_combs = combs['comb_id'].unique()
        for comb_id in uniq_combs:
            exp_descriptor = f'g-{grp_id}__c-{comb_id}'
            print(exp_descriptor)
            runs = combs.loc[(combs['comb_id'] == comb_id)]
            param_set = runs.values[0,-1]
            param_dict = ast.literal_eval(param_set)
            # gather replaced parameters for experiment here
            if parser_args.select_yaml_fields:
                select_args = [f'--{arg}' for arg in parser_args.select_yaml_fields]
                to_del_keys = set(param_dict.keys()) - set(select_args)
                for key in to_del_keys: param_dict.pop(key, None)
            global_param_set.update({exp_descriptor : param_dict})
            try:
                for exp in runs.values:
                    exp_file_name = exp[2]
                    exp_run_path = os.path.join(parser_args.results_dir, exp_file_name, 'metrics', f'{exp_file_name}{parser_args.custom_str}.csv')
                    print(exp_run_path)
                    if os.path.isfile(exp_run_path):  
                        cols = list(pd.read_csv(exp_run_path, nrows=1)) # read in columns
                        _cols = []
                        for f_m in parser_args.filter_metrics:
                            for col in cols:
                                if f_m in col: _cols.append(col)
                        if len(_cols) > len(col_data): col_data = _cols
            except Exception as ex: traceback.print_exc(ex)

    index_metrics = ['avg', 'std'] # FIXME: adapt if adding more than avg./std.
    #index_metrics = ['avg'] # TODO: selective metrics, add things like min/max/quantiles etc.
    print(col_data)
    prod_tuples = list(itertools.product(col_data, index_metrics))
    col_header = pd.MultiIndex.from_tuples(
        prod_tuples,
        names=['test_task', 'descriptor']
    )
    with open(f'{parser_args.save_path}/params.yaml', 'w') as f_s:
        yaml.dump(global_param_set, f_s)

    # ------------------------------------------ GATHER GRP RESULTS
    global_df = None
    for grp_id in selected_groups:
        combs = db_file.loc[(db_file['group_id'] == grp_id)]
        uniq_combs = combs['comb_id'].unique()
        for comb_id in uniq_combs:
            exp_descriptor = f'g-{grp_id}__c-{comb_id}'
            runs = combs.loc[(combs['comb_id'] == comb_id)]            
            
            group_results = pd.DataFrame()
            try:
                cols = list()
                for i, exp in enumerate(runs.values):
                    exp_file_name = exp[2]
                    exp_run_path = os.path.join(parser_args.results_dir, exp_file_name, 'metrics', f'{exp_file_name}{parser_args.custom_str}.csv')
                    #print(exp_run_path)
                    if os.path.isfile(exp_run_path):
                        if i == 0:
                            cols = list(pd.read_csv(exp_run_path, nrows=1))
                            exp_run_results = pd.read_csv(
                                exp_run_path, 
                                usecols=col_data,
                            )
                        else:
                            exp_run_results = pd.read_csv(
                                exp_run_path, 
                                usecols=col_data,
                                header=0,
                            )
                        group_results = pd.concat([group_results, exp_run_results], ignore_index=True)
                        #print(group_results)
                # ------------------------------------------ START GLOBAL EVAL
                meaned_series = pd.Series.mean(group_results)
                meaned_series_val = np.round(meaned_series.values, decimals=4)
                #print(meaned_series)
                #FIXME: add if needed: pd.Series.min(group_results), pd.Series.max(group_results), pd.Series.median(group_results)
                stddev_series_val = np.round(pd.Series.std(group_results, axis=0).values, decimals=4)
                transformed = np.vstack((meaned_series_val, stddev_series_val)).T.flatten() # TODO: selective metrics, add things like min/max/quantiles etc.
                #transformed = meaned_series_val.T.flatten()    # FIXME: comment in if only printing means...
                
                transformed = np.expand_dims(transformed, axis=0)
                #print(transformed.shape)
                #print(transformed)
                #print(col_header[:transformed.shape[1]])
                print([exp_descriptor])
                # concat DFs
                if global_df is None:
                    global_df = pd.DataFrame(
                        data=transformed,
                        columns=col_header[:transformed.shape[1]],
                        index=[exp_descriptor],
                    )
                else:
                    temp_df = pd.DataFrame(
                        data=transformed,
                        columns=col_header[:transformed.shape[1]],
                        index=[exp_descriptor],
                    )
                    global_df = pd.concat([global_df, temp_df])
            except Exception as ex: traceback.print_exc(ex)
    #print(global_df)
    global_df.to_csv(f'{parser_args.save_path}/metrics.csv')
    
    json_dict = {}
    for series_name, series in global_df.items():
        if series_name[1] == 'avg':
            json_dict.update({series_name[0] : series[0]})
        
    #print(json_dict)
    with open(f'{parser_args.save_path}/metrics.json', 'w') as f_s:
        json.dump(json_dict, f_s)
