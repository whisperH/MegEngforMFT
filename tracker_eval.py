import os
import sys
import argparse
import motmetrics as mm
import numpy as np
import pandas as pd

# PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
# DATA_PATH = os.path.join(PROJECT_PATH, 'data/')


class TrackEvaluator(object):
    def __init__(self, args, data_type='track', maxDist=450, w_MOTA=1, w_IDF1=1):
        self.maxDist = maxDist
        self.data_path = args.data_path
        self.dataset_type = args.dataset_type
        self.track_result_path = args.track_result_path
        self.gt_path = args.gt_path
        if self.track_result_path is not None and self.gt_path is not None:
            self.eval_datapath = self.getTrackDataPath(self.track_result_path, frame_fre='1')
            self.gt_datapath = self.getTrackDataPath(self.gt_path, frame_fre='1')

            self.eval_5_datapath = self.getTrackDataPath(self.track_result_path, frame_fre='5')
            self.gt_5_datapath = self.getTrackDataPath(self.gt_path, frame_fre='5')

        # else:
        #     self.gt_datapath = self.getEvalDataPath(
        #         data_path=args.data_path,
        #         data_type='gt', eva_type='val_half',
        #         frame_fre='1', dataset_type=self.dataset_type
        #     )
        #     self.eval_datapath = self.getEvalDataPath(
        #         data_path=args.data_path,
        #         data_type=data_type, eva_type='val_half',
        #         frame_fre='1', dataset_type=self.dataset_type
        #     )
        #     self.gt_5_datapath = self.getEvalDataPath(
        #         data_path=args.data_path,
        #         data_type='gt', eva_type='val_half',
        #         frame_fre='5', dataset_type=self.dataset_type
        #     )
        #     self.eval_5_datapath = self.getEvalDataPath(
        #         data_path=args.data_path,
        #         data_type=data_type, eva_type='val_half',
        #         frame_fre='5', dataset_type=self.dataset_type
        #     )
        self.w_MOTA = w_MOTA
        self.w_IDF1 = w_IDF1

    def run(self):
        '''
        :return:
        '''
        scores = []
        for gt_, track_ in zip(sorted(self.gt_datapath), sorted(self.eval_datapath)):
            print("starting to evaluate:{} - {}".format(gt_, track_))
            save_path = os.path.split(track_)[0]
            try:
                score = self.MOT_Evaluation(gt_, track_, self.maxDist, save_path)
            except Exception as e:
                print(e)
                score = 0
            scores.append(score)

        scores_5 = []
        for gt_, track_ in zip(sorted(self.gt_5_datapath), sorted(self.eval_5_datapath)):
            print("starting to evaluate:{} - {}".format(gt_, track_))

            save_path = os.path.split(track_)[0]
            try:
                score = self.MOT_Evaluation(gt_, track_, self.maxDist, save_path)
            except Exception as e:
                print(e)
                score = 0

            scores_5.append(score)
        return scores, scores_5

    def getTrackDataPath(self, datapath, frame_fre='1'):
        eval_datapath = []
        files = os.listdir(
            os.path.join(datapath)
        )
        for file in files:
            if 's'+frame_fre in file:
                eval_datapath.append(
                    os.path.join(
                        datapath,
                        file
                    )
                )
        return eval_datapath

    # def getEvalDataPath(self, data_path, eva_type, data_type='gt', frame_fre='5', dataset_type='preliminary'):
    #     '''
    #     get data path needed to be evaluated
    #     :param dataset_type: 'intermediary' or 'preliminary'
    #     :param data_type: 'gt' or 'track'
    #     :param eva_type: 'val_half' or 'test'
    #     :param frame_fre: '1': no selected-frame; '5' select 5 frame each sub-dataset
    #     :return: <list> evaluated data path
    #     '''
    #     eval_datapath = []
    #     seqs = os.listdir(
    #         os.path.join(data_path, dataset_type)
    #     )
    #     for seq in seqs:
    #         eval_datapath.append(
    #             os.path.join(
    #                 data_path, dataset_type, seq, 'gt',
    #                 '{}_{}_{}.txt'.format(data_type, frame_fre, eva_type)
    #             )
    #         )
    #     return eval_datapath

    def MOT_Evaluation(
            self, gt_filepath, track_filepath,
            maxDist, outputDir
    ):
        """

        Given a ground annotated dataset and a set of predictions,
        calcualtes the full suite of MOT metriccs + the MTBF metrics

        Input:
            track_filepath: Path to tracking CSV file
            gt_filepath: Path to ground truth CSV file
            maxDist: Distance threshold
            outputDir: Path to where the evaluation output file should be saved
        """

        # gt中bbox的标注格式是 xywh
        gt_df = pd.read_csv(
            gt_filepath, sep=",",
        )
        gt_df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "cat", "iscrowd"]

        # gt中bbox的标注格式是 xyxy,
        det_df = pd.read_csv(
            track_filepath, sep=","
        )
        det_df.columns = ["frame", "id", "bb_left", "bb_top", "bb_right", "bb_bottom", "conf", "cat", "iscrowd"]

        posFunc = get2Dpos
        distFunc = pairwiseDistance

        gt_frame_col = "frame"
        det_frame_col = "frame"

        gt_df["cam1_x_bb"] = gt_df["bb_left"] + gt_df["bb_width"] // 2
        gt_df["cam1_y_bb"] = gt_df["bb_top"] + gt_df["bb_height"] // 2
        gt_df = gt_df.dropna(subset=["cam1_x_bb", "cam1_y_bb"])

        det_df["cam1_x_bb"] = (det_df["bb_left"] + det_df["bb_right"]) // 2
        det_df["cam1_y_bb"] = (det_df["bb_top"] + det_df["bb_bottom"]) // 2
        det_df = det_df.dropna(subset=["cam1_x_bb", "cam1_y_bb"])

        gt_frames = gt_df[gt_frame_col].unique()
        det_frames = det_df[det_frame_col].unique()

        gt_frames = [int(x) for x in gt_frames]
        det_frames = [int(x) for x in det_frames]

        frames = list(set(gt_frames + det_frames))

        print("Amount of GT frames: {}\nAmount of det frames: {}\nSet of all frames: {}".format(len(gt_frames),
                                                                                                len(det_frames),
                                                                                                len(frames)))
        assert len(gt_frames) == len(det_frames), 'missing some frames in det_files'
        acc = mm.MOTAccumulator(auto_id=False)

        for frame in frames:
            # print("processing  frame:{}".format(frame))

            # Get the df entries for this specific frame
            gts = gt_df[gt_df[gt_frame_col] == frame]
            dets = det_df[det_df[det_frame_col] == frame]

            gt_data = True
            det_data = True

            # Get ground truth positions, if any
            if len(gts) > 0:
                gt_pos, gt_ids = posFunc(gts)
                # gt_ids = ["gt_{}".format(x) for x in gt_ids]
            else:
                gt_ids = []
                gt_data = False

            # Get detections, if any

            if len(dets) > 0:
                det_pos, det_ids = posFunc(dets)
                # det_ids = ["det_{}".format(x) for x in det_ids]
            else:
                det_ids = []
                det_data = False

            # Get the L2 distance between ground truth positions, and the detections
            if gt_data and det_data:
                dist = distFunc(gt_pos, det_pos, maxDist=maxDist).tolist()
            else:
                dist = []

            # Update accumulator
            acc.update(gt_ids,  # Ground truth objects in this frame
                       det_ids,  # Detector hypotheses in this frame
                       dist,  # Distance between ground truths and observations
                       frame)

        metrics = calcMetrics(acc)

        writeMOTtoTXT(metrics, os.path.join(outputDir, "eval_res_{}".format(maxDist)))

        Score = EvalScore(
            metrics["mota"].values[0],
            metrics["idf1"].values[0],
            self.w_MOTA, self.w_IDF1
        )
        return Score

def EvalScore(MOTA, IDF1, w_MOTA, w_IDF1):
    '''
    计算 MOTA与IDF1的加权调和平均数
    :param MOTA:
    :param IDF1:
    :param w_MOTA:
    :param w_IDF1:
    :return:
    '''
    numerator = (MOTA + IDF1) * (w_MOTA + w_IDF1)
    denominator = (MOTA * w_IDF1) + (IDF1 * w_MOTA)
    return numerator / denominator

def get2Dpos(df):
    """
    Returns the 2D position in a dataset. Depending on the set task it either returns the positon of the designated keypoint or of the bounding box center

    Input:
        df: Pandas dataframe
        cam: Camera view
        task: if 'bbox' return the bbox center, else return the keypoint x and y coordinates

    Output:
        pos: Numpy array of size [n_ids, 3] containing the 3d position
        ids: List of IDs
    """
    ids = df["id"].unique()
    ids = [int(x) for x in ids]

    pos = np.zeros((len(ids), 2))
    for idx, identity in enumerate(ids):
        df_id = df[df["id"] == identity]
        pos[idx, 0] = df_id["cam1_x_bb"]
        pos[idx, 1] = df_id["cam1_y_bb"]

    return pos, ids


def pairwiseDistance(X, Y, maxDist):
    """
    X and Y are n x d and m x d matrices, where n and m are the amount of observations, and d is the dimensionality of the observations
    """

    X_ele, X_dim = X.shape
    Y_ele, Y_dim = Y.shape

    assert X_dim == Y_dim, "The two provided matrices not have observations of the same dimensionality"

    mat = np.zeros((X_ele, Y_ele))

    for row, posX in enumerate(X):
        for col, posY in enumerate(Y):
            mat[row, col] = np.linalg.norm(posX - posY)

    mat[mat > maxDist] = np.nan

    return mat


def writeMOTtoTXT(metrics, output_file):
    '''
    Writes the provided MOT + MTBF metrics results for a given cateogry to a txt file

    Input:
        - metrics: pandas DF containing the different metrics
        - output-file: Path to the output file
    '''

    with open(output_file, "w") as f:
        f.write("MOTA = {:.1%}\n".format(metrics["mota"].values[0]))
        f.write("MOTAL = {:.1%}\n".format(metrics["motal"].values[0]))
        f.write("MOTP = {:.3f}\n".format(metrics["motp"].values[0]))
        f.write("Precision = {:.1%}\n".format(metrics["precision"].values[0]))
        f.write("Recall = {:.1%}\n".format(metrics["recall"].values[0]))
        f.write("ID Recall = {:.1%}\n".format(metrics["idr"].values[0]))
        f.write("ID Precision = {:.1%}\n".format(metrics["idp"].values[0]))
        f.write("ID F1-score = {:.1%}\n".format(metrics["idf1"].values[0]))
        f.write("Mostly Tracked = {:d}\n".format(metrics["mostly_tracked"].values[0]))
        f.write("Partially Tracked = {:d}\n".format(metrics["partially_tracked"].values[0]))
        f.write("Mostly Lost = {:d}\n".format(metrics["mostly_lost"].values[0]))
        f.write("False Positives = {:d}\n".format(metrics["num_false_positives"].values[0]))
        f.write("False Negatives = {:d}\n".format(metrics["num_misses"].values[0]))
        f.write("Identity Swaps = {:d}\n".format(metrics["num_switches"].values[0]))
        f.write("Fragments = {:d}\n".format(metrics["num_fragmentations"].values[0]))
        f.write("MTBF-std = {:.3f}\n".format(metrics["mtbf_std"].values[0]))
        f.write("MTBF-mono = {:.3f}\n".format(metrics["mtbf_mono"].values[0]))
        f.write("Total GT points = {:d}\n".format(metrics["num_objects"].values[0]))
        f.write("Total Detected points = {:d}\n".format(metrics["num_predictions"].values[0]))
        f.write("Number of frames = {:d}\n".format(metrics["num_frames"].values[0]))
        f.write("\n\n")

    header = ["MOTA", "MOTAL", "MOTP", "Precision", "Recall", "ID Recall", "ID Precision", "ID F1-score",
              "Mostly Tracked", "Partially Tracked", "Mostly Lost", "False Positives", "False Negatives",
              "Identity Swaps", "Fragments", "MTBF-std", "MTBF-mono"]
    value = ["{:.1%}".format(metrics["mota"].values[0]),
             "{:.1%}".format(metrics["motal"].values[0]),
             "{:.3f}".format(metrics["motp"].values[0]),
             "{:.1%}".format(metrics["precision"].values[0]),
             "{:.1%}".format(metrics["recall"].values[0]),
             "{:.1%}".format(metrics["idr"].values[0]),
             "{:.1%}".format(metrics["idp"].values[0]),
             "{:.1%}".format(metrics["idf1"].values[0]),
             "{:d}".format(metrics["mostly_tracked"].values[0]),
             "{:d}".format(metrics["partially_tracked"].values[0]),
             "{:d}".format(metrics["mostly_lost"].values[0]),
             "{:d}".format(metrics["num_false_positives"].values[0]),
             "{:d}".format(metrics["num_misses"].values[0]),
             "{:d}".format(metrics["num_switches"].values[0]),
             "{:d}".format(metrics["num_fragmentations"].values[0]),
             "{:.3f}".format(metrics["mtbf_std"].values[0]),
             "{:.3f}".format(metrics["mtbf_mono"].values[0])]

    with open(output_file[:-4] + ".tex", "w") as f:
        f.write(" & ".join(header) + "\n")
        f.write(" & ".join(value))

    with open(output_file[:-4] + ".csv", "w") as f:
        f.write(";".join(header) + "\n")
        f.write(";".join(value))


def calcMetrics(acc):
    """
    Calcualtes all the relevant metrics for the dataset for the specified task

    Input:
        acc: MOT Accumulator oject
        task: String designating the task which has been evaluated

    Output:
        summary: Pandas dataframe containing all the metrics
    """
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc],
        metrics=mm.metrics.motchallenge_metrics + ["num_objects"] + ["num_predictions"] + ["num_frames"],
        names=['top'])

    summary["motal"] = MOTAL(summary)
    mtbf_metrics = MTBF(acc.mot_events)
    summary["mtbf_std"] = mtbf_metrics[0]
    summary["mtbf_mono"] = mtbf_metrics[1]
    print(summary)

    formatter = {**mh.formatters, "num_objects": '{:d}'.format, "num_predictions": '{:d}'.format,
                 "num_frames": "{:d}".format, "motal": '{:.1%}'.format, "mtbf_std": '{:.3f}'.format,
                 "mtbf_mono": '{:.3f}'.format}
    namemap = {**mm.io.motchallenge_metric_names, "num_objects": "# GT", "num_predictions": "# Dets",
               "num_frames": "# Frames", "motal": "MOTAL", "mtbf_std": "MTBF-std", "mtbf_mono": "MTBF-mono"}

    strsummary = mm.io.render_summary(
        summary,
        formatters=formatter,
        namemap=namemap
    )
    print(strsummary)
    print(acc.mot_events)

    return summary


def MOTAL(metrics):
    """
    Calcualtes the MOTA variation where the amount of id switches is attenuated by using hte log10 function
    """
    return 1 - (metrics["num_misses"] + metrics["num_false_positives"] + np.log10(metrics["num_switches"] + 1)) / \
           metrics["num_objects"]


def MTBF(events):
    """
    Calclautes the Mean Time Betwwen Failures (MTBF) Metric from the motmetric events dataframe

    Input:
        events: Pandas Dataframe structured as per the motmetrics package

    Output:
        MTBF_standard: The Standard MTBF metric proposed in the original paper
        MTBF_monotonic: The monotonic MTBF metric proposed in the original paper
    """

    unique_gt_ids = events.OId.unique()
    seqs = []
    null_seqs = []
    for gt_id in unique_gt_ids:
        gt_events = events[events.OId == gt_id]

        counter = 0
        null_counter = 0

        for _, row in gt_events.iterrows():
            if row["Type"] == "MATCH":
                counter += 1
            elif row["Type"] == "SWITCH":
                seqs.append(counter)
                counter = 1
            else:
                seqs.append(counter)
                counter = 0
                null_counter = 1

            if counter > 0:
                if null_counter > 0:
                    null_seqs.append(null_counter)
                    null_counter = 0

        if counter > 0:
            seqs.append(counter)
        if null_counter > 0:
            null_seqs.append(null_counter)

    seqs = np.asarray(seqs)
    seqs = seqs[seqs > 0]

    if len(seqs) == 0:
        return (0, 0)
    else:
        return (sum(seqs) / len(seqs), sum(seqs) / (len(seqs) + len(null_seqs)))


class TestUnit():
    def __init__(self):
        pass

    def gtInput(self):
        scores, scores_5 = TrackEvaluator(data_type='gt').run()
        print('original dataset get scores:', np.mean(scores))
        print('selected-5-frames dataset get scores:', np.mean(scores_5))

    def trackInput(self):
        scores, scores_5 = TrackEvaluator(data_type='track').run()
        print('original dataset get scores:', np.mean(scores))
        print('selected-5-frames dataset get scores:', np.mean(scores_5))

    def zeroInput(self):
        scores, scores_5 = TrackEvaluator(data_type='zero').run()
        print('original dataset get scores:', np.mean(scores))
        print('selected-5-frames dataset get scores:', np.mean(scores_5))

    def nofileInput(self):
        scores, scores_5 = TrackEvaluator(data_type='xxxx').run()
        print('original dataset get scores:', np.mean(scores))
        print('selected-5-frames dataset get scores:', np.mean(scores_5))

    def run(self):
        print("test for zero input")
        self.zeroInput()

        print("test for no input file")
        self.nofileInput()

        print("test for gt input")
        self.gtInput()

        print("test for track input")
        self.trackInput()

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track_result_path",
        default='/home/huangjinze/code/MegEngTrack/data/submit',
        type=str, help="track_result filepath"
    )
    parser.add_argument(
        "--gt_path",
        default='/home/huangjinze/code/MegEngTrack/data/interal',
        type=str, help="ground truth filepath"
    )
    parser.add_argument(
        "--data_path",
        default="/home/megstudio/workspace/data/",
        type=str, help="submit filepath"
    )
    parser.add_argument(
        "--dataset_type",
        default="train",
        type=str, help="'intermediary' or 'preliminary'"
    )

    return parser

if __name__ == "__main__":
    '''
    track_result_path: 参赛者上传的存储结果文件夹
    '''
    # TestUnit().run()
    parser = make_parser()
    args = parser.parse_args()
    scores, scores_5 = TrackEvaluator(
        args,
        data_type='track'
    ).run()
    print('original dataset get scores:', scores)
    print('original dataset get scores:', np.mean(scores))
    print('selected-5-frames dataset get scores:', scores_5)
    print('selected-5-frames dataset get scores:', np.mean(scores_5))

