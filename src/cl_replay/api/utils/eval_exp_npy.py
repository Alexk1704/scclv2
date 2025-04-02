import os
import tarfile
import traceback
import numpy as np

from argparse import ArgumentParser
from pathlib import Path



if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--results_dir", required=True, type=str, help="rel. path to the parent directory holding exp. results.")
    
    args = parser.parse_args()
    
    def unpack_archive(file_path):
        def check_existence(tar, dest_path):
            for tarinfo in tar:
                file_path = os.path.join(dest_path, tarinfo.name)
                if os.path.exists(file_path):
                    return True
            return False
        if str(file_path).endswith('tar.gz'):
            if tarfile.is_tarfile(file_path):
                tar = tarfile.open(file_path)
                exp_id = str(file_path).split('/', 5)[-1][:-7]
                parent_dir = os.path.dirname(file_path)
                save_dir = os.path.join(parent_dir, exp_id)
                print(parent_dir, exp_id)
                # with tarfile.open(file_path) as tar:
                #     for tarinfo in tar.getmembers():
                #         if tarinfo.name.endswith(exp_id):
                #             ex = tarinfo
                #             tar.extract(member=ex, path=save_dir)
                if not check_existence(tar, save_dir):
                    tar.extractall(path=save_dir)
                    tar.close()
        # else:
        #     with open(file_path, 'rb') as fp:
        #         with tarfile.open(fileobj=fp, mode='r') as tar:
        #             if not check_existence(tar, os.path.dirname(file_path)):
        #                 tar.extractall(path=os.path.dirname(file_path))
    
    path_to_results = Path(args.results_dir)
    if Path.exists(path_to_results):
        if Path.is_dir(path_to_results):
            for file in path_to_results.iterdir():
                try:
                    unpack_archive(file)
                except Exception as ex:
                    print(traceback.format_exc())

    acc_paths = []
    for file in path_to_results.iterdir():
        if Path.is_dir(file):
            metrics_path = os.path.join(file, 'tmp/results/')
            for file in Path(metrics_path).iterdir():
                metrics_dir = Path(os.path.join(file, 'metrics'))
                if Path.is_dir(metrics_dir):
                    #path = metrics_dir.rglob('*_accmat.npy')
                    for file in metrics_dir.iterdir():
                        if str(file).endswith('accmat.npy'):
                            acc_paths.append(file)

    results_dict = {}
    for path in acc_paths:
        path_str = str(path)
        id_str = path_str.split('/', 6)[5]  # NOTE: adjust no. of splits
        print(path_str)
        print(id_str)
        id_ = id_str.split('__', 1)[1]
        ids = id_.split('__', 2)
        g_id = int(ids[0].split('-', 1)[1])
        c_id = int(ids[1].split('-', 1)[1])
        r_id = int(ids[2].split('-', 1)[1]) 
        acc_mat = np.load(path_str)
        
        if g_id in results_dict:
            entry = results_dict[g_id]
            if c_id in entry:
                entry = entry[c_id]
                entry.update({r_id : acc_mat})
            else:
                entry.update({c_id : { r_id : acc_mat}})
        else:
            results_dict.update({g_id : {c_id : { r_id : acc_mat}}})

    np.set_printoptions(formatter={'float':'{:.2f}'.format})
    
    score_dict = {}
    for group in results_dict.keys():
        score_dict.update({group : {}})
    
    for group in sorted(results_dict.keys()):
        for combinatorial in sorted(results_dict[group]):
            acc_mats = []
            for run in results_dict[group][combinatorial]:
                acc_mats.append(results_dict[group][combinatorial][run])
            mean = np.mean(acc_mats, 0)
            std = np.std(acc_mats, 0)

            # if group == 0:
            print("-------")
            print(f"G: {group} C: {combinatorial}")
            print(mean)
            # print(std)
            
            if mean.shape[0] == 7: # separate A
                print('\n')
                forg_score = (mean[6][1] + mean[6][3] + mean[6][5]) / 3
                
                max_task_acc = np.max(mean, axis=0)  # get max acc. over col (task)
                
                f_t1 = max_task_acc[1] - mean[6][1];
                f_t2 = max_task_acc[3] - mean[6][3]
                f_t3 = max_task_acc[5] - mean[6][5]
                
                print(f'forgetting T2: {f_t1}')
                print(f'forgetting T4: {f_t2}')
                print(f'forgetting T6: {f_t3}')
                
                print(f'accumulated forgetting: {(f_t1+f_t2+f_t3)/3}')
                
                # print('long f_t acc: {:.2f},{:.2f},{:.2f}'.format(mean[6][1], mean[6][3], mean[6][5])) 
                # print('long f_t score: {:.2f}'.format(forg_score))
                
                immediate_forg_score = (mean[2][1] + mean[4][3] + mean[6][5]) / 3
                print('immediate f_t (paired): {:.2f} -> {:.2f} / {:.2f} -> {:.2f} / {:.2f} -> {:.2f}'.format(mean[1][1], mean[2][1], mean[3][3], mean[4][3], mean[5][5], mean[6][5]))
                print('immediate f_t (dedicated): {:.2f} -> {:.2f} / {:.2f} -> {:.2f} / {:.2f} -> {:.2f}'.format(mean[0][1], mean[1][1], mean[2][3], mean[3][3], mean[4][5], mean[5][5]))
                # print('immediate f: {:.2f},{:.2f},{:.2f}'.format((mean[1][1] - mean[2][1]), (mean[3][3] - mean[4][3]), (mean[5][5] - mean[6][5])))   # 1st/2nd/3rd forgetting task acc. (should be as low as possible)
                # print('immediate f score: {:.2f}'.format(immediate_forg_score))
                
                print('retain set acc: {:.2f}'.format(mean[6][7]))   # retain set acc after complete training (should be as high as possible)

            if mean.shape[0] == 8: # separate B
                print('\n')
                
                forg_score = (mean[7][2] + mean[7][4] + mean[7][6]) / 3
                
                max_task_acc = np.max(mean, axis=0)  # get max acc. over col (task)
                
                f_t1 = max_task_acc[2] - mean[7][2]
                f_t2 = max_task_acc[4] - mean[7][4]
                f_t3 = max_task_acc[6] - mean[7][6]

                print(f'forgetting T3: {f_t1}')
                print(f'forgetting T5: {f_t2}')
                print(f'forgetting T7: {f_t3}')
                
                print(f'accumulated forgetting: {(f_t1+f_t2+f_t3)/3}')
                
                # print('long f_t acc: {:.2f},{:.2f},{:.2f}'.format(mean[7][2], mean[7][4], mean[7][6])) 
                # print('long f_t score: {:.2f}'.format(forg_score))
                
                immediate_forg_score = (mean[3][2] + mean[5][4] + mean[7][6]) / 3
                print('immediate f_t (paired): {:.2f} -> {:.2f} / {:.2f} -> {:.2f} / {:.2f} -> {:.2f}'.format(mean[2][2], mean[3][2], mean[4][4], mean[5][4], mean[6][6], mean[7][6]))
                print('immediate f_t (dedicated): {:.2f} -> {:.2f} / {:.2f} -> {:.2f} / {:.2f} -> {:.2f}'.format(mean[1][2], mean[2][2], mean[3][4], mean[4][4], mean[5][6], mean[6][6]))
                # print('immediate f_t acc: {:.2f},{:.2f},{:.2f}'.format((mean[2][2] - mean[3][2]), (mean[4][4] - mean[5][4]), (mean[6][6] - mean[7][6])))   # 1st/2nd/3rd forgetting task acc. (should be as low as possible)
                # print('immediate f_t score: {:.2f}'.format(immediate_forg_score))
                
                print('retain set acc: {:.2f}'.format(mean[7][8]))   # retain set acc after complete training (should be as high as possible)

            # if mean.shape[0] == 4: # mixed 
            #     forg_score = (mean[1][0] + mean[2][0] + mean[3][0]) / 3
            #     print('task f: {:.2f},{:.2f},{:.2f}'.format(mean[1][0], mean[2][0], mean[3][0]))
            #     print('f score: {:.2f}'.format(forg_score))
                
            #     print('retain set acc: {:.2f}'.format(mean[3][4]))   # retain set acc after complete training (should be as high as possible)
                
            # update score dict with accumulated forgetting values (accuracy in this case)
            score_dict[group].update({combinatorial : forg_score })
                
    
    # for group in score_dict:
    #     if group == 0:
    #         sorted_ = sorted(score_dict[group].items(), key=lambda item: item[1])
    #         top_ = sorted(sorted_[:48])
    #         print(top_)
                
                
