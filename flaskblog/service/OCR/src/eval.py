from difflib import SequenceMatcher
from collections import OrderedDict
from pathlib import Path
import numpy as np

import pandas as pd

def get_acc(a, b):
    return SequenceMatcher(None, a, b).ratio()


def eval_result(true_dict:dict, pred_dict:dict)->dict:
    
    if true_dict is None: # label is not available
        true_dict = {k:'-' for k in pred_dict.keys()}
        has_label = False
    else:
        has_label = True
    
    assert true_dict.keys() == pred_dict.keys(), '2 dictionaries must have same keys'
    
    eval_dict = OrderedDict()
    sum_acc = 0
    for k, true_val in true_dict.items():
        pred_val = pred_dict[k]
        
        true_dict_, pred_dict_ = true_val.lower(), pred_val.lower()
        
        if has_label:
            acc = get_acc(true_dict_,pred_dict_)
            sum_acc += acc
        else:
            acc = '-'
    
        eval_dict[f'{k}_true'] = true_val
        eval_dict[f'{k}_pred'] = pred_val
        eval_dict[f'{k}_acc']  = acc

    if has_label:
        eval_dict['avg_by_img'] = sum_acc / len(true_dict)
    else:
        eval_dict['avg_by_img'] = '-'
        
    return eval_dict


def eval_result_no_text(true_dict:dict, pred_dict:dict)->dict:
    """ 
    ! not being used
    """
    
    eval_dict_with_text = eval_result(true_dict, pred_dict)
    eval_dict_no_text   = dict(filter(lambda kv: 'acc' in kv[0], 
                                      eval_dict_with_text.items()))
    return eval_dict_no_text


################################################################################
################################################################################
##        #####  #   #  #####  #####  ####   #####    #     ####  #####       ##
##          #    ##  #    #    #      #   #  #       # #   #      #           ##
##          #    # # #    #    #####  ####   ####   #####  #      #####       ##
##          #    #  ##    #    #      # #    #      #   #  #      #           ##
##        #####  #   #    #    #####  #  ##  #      #   #   ####  #####       ##
################################################################################
################################################################################


def get_eval_df(true_dicts:list, pred_dicts:list, less=1, file_paths:list=None):
    """
    true_dicts, pred_dicts: list of dictionary/json, from parsed images
    less: integer
        0: return full Dataframe with all texts
        1: return DataFrame with accuracy only
        2: return "avg_by_field" row
        3: return "avg_by_img" column
        4: return average scalar
    file_paths: replace raw index with filename from file_path
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    if not isinstance(true_dicts, list):
        true_dicts = [true_dicts]
    if not isinstance(pred_dicts, list):
        pred_dicts = [pred_dicts]
        
    assert len(true_dicts) == len(pred_dicts), '2 lists must have same lengths'
    
    eval_dicts = []
    
    for ix, true_dict in enumerate(true_dicts):
        pred_dict = pred_dicts[ix]
        eval_dicts.append(eval_result(true_dict=true_dict,
                                      pred_dict=pred_dict))
        
    df  = pd.DataFrame(eval_dicts)
    df2 = df.replace('-', np.nan)
    df.loc['avg_by_field'] = df2.mean(axis=0)
    df.fillna('-', inplace=True)
    
    if file_paths is not None:
        l = list(map(lambda path: Path(path).stem, file_paths))
        d = dict(zip(range(len(l)),l))
        df.rename(index=d,inplace=True)
    
    if less == 0 :
        return df
    
    # remove columns which is not "acc"
    cols = [col for col in df.columns if 'acc' in col ] + [df.columns[-1]]
    df   = df[cols]
    
    if less == 1:
        return df
    
    elif less == 2:
        series = df.iloc[-1]
        series.rename({"avg_by_img":"average"},inplace=True)
        return series
        
    elif less == 3:
        series = df.iloc[:,-1]
        series.rename({"avg_by_field":"average"},inplace=True)
        return series
    
    elif less == 4:
        scalar = df.iloc[-1,-1]
        return scalar