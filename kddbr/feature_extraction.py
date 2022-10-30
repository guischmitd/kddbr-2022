import pandas as pd
import numpy as np
import cv2
from context import RAWDATADIR
from helpers import read_before_and_after

def add_lags(
    df, 
    n_lags=3,
    lag_cols = [
        'Altitude', 
        'Delta', 
        'east_median',
        'north_median',
        'east_median_sift', 
        'north_median_sift', 
        'n_matches', 
        'n_matches_sift'
        ]):

    Xy = df.copy()

    for i in range(1, n_lags + 1):
        lag_feats = df.groupby('sequence')[lag_cols].shift(i)
        leap_feats = df.groupby('sequence')[lag_cols].shift(-i)
        diff_feats = df.groupby('sequence')[lag_cols].diff(i)
        leap_diff_feats = df.groupby('sequence')[lag_cols].diff(-i)

        Xy = Xy.join(lag_feats, rsuffix=f'_lag_{i}')
        Xy = Xy.join(leap_feats, rsuffix=f'_leap_{i}')
        
        Xy = Xy.join(diff_feats, rsuffix=f'_diff_{i}')
        Xy = Xy.join(leap_diff_feats, rsuffix=f'_leap_diff_{i}')

    return Xy.fillna(0.0)

class ImageFeatureExtractor:
    def __init__(self, train_dir=RAWDATADIR / 'train/train', test_dir=RAWDATADIR / 'test/test') -> None:
        self.train_dir = train_dir
        self.test_dir = test_dir

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.shitomasi_params = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 7,
            blockSize = 7 )

        self.lk_params = dict(
            winSize  = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

        self.sift = cv2.SIFT_create()

    def _apply_clahe_multichannel(self, img):
        """Expects [h, w, c]"""
        processed = []
        for c_index in range(img.shape[-1]):
            processed.append(self.clahe.apply(img[:, :, c_index]))
        
        return np.dstack(processed)

    def _get_lk_features(self, img1, img2, apply_clahe=True):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if apply_clahe:
            gray1, gray2 = self.clahe.apply(gray1), self.clahe.apply(gray2)
            
        p0 = cv2.goodFeaturesToTrack(gray1, mask = None, **self.shitomasi_params)

        if p0 is None:
            return None

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if np.sum(st) == 0:
            return None

        diffs = []
        centroids = []
        for _, (new, old) in enumerate(zip(good_new, good_old)):
            point1 = old.ravel()
            point2 = new.ravel()

            diff = point2 - point1

            # Flip for North/East axis alignment with X, Y on image
            diff[0] = - diff[0]
            diffs.append(point2 - point1)
            
            centroid = (point1 + point2) / 2
            centroids.append(centroid)

        mean, median, std = np.mean(diffs, axis=0), np.median(diffs, axis=0), np.std(diffs, axis=0)
        c_mean = np.mean(centroids, axis=0)
        
        feats = {
            'east_mean': mean[0], 'east_median': median[0], 'east_std': std[0],
            'north_mean': mean[1], 'north_median': median[1], 'north_std': std[1],
            'centroid_east': 60 - c_mean[0], 'centroid_north': 60 - c_mean[1], 'n_matches': np.sum(st),
            }

        return pd.Series(feats)

    def _get_sift_features(self, img1, img2):
        
        img1, img2 = self._apply_clahe_multichannel(img1), self._apply_clahe_multichannel(img2)
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        # Check keypoint detection
        if len(kp1) <=1 or len(kp2) <= 1:
            return None

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i]=[1,0]
        
        # Check match number
        masked_matches = np.asarray(matches)[np.asanyarray(matchesMask) == 1]
        if len(masked_matches) == 0:
            return None
        
        diffs = []
        centroids = []
        pts1 = []
        pts2 = []
        
        for i, m in enumerate(masked_matches):
            point2 = np.asarray(kp2[m.trainIdx].pt)
            point1 = np.asarray(kp1[m.queryIdx].pt)
            
            diff = point2 - point1
            pts1.append(point1)
            pts2.append(point2)

            # Flip for North/East axis alignment with X, Y on image
            diff[0] = - diff[0]
            diffs.append(point2 - point1)
            
            centroid = (point1 + point2) / 2
            centroids.append(centroid)

        mean, median, std = np.mean(diffs, axis=0), np.median(diffs, axis=0), np.std(diffs, axis=0)
        c_mean = np.mean(centroids, axis=0)

        feats = {
            'east_mean_sift': mean[0], 'east_median_sift': median[0], 'east_std_sift': std[0],
            'north_mean_sift': mean[1], 'north_median_sift': median[1], 'north_std_sift': std[1],
            'centroid_east_sift': 60 - c_mean[0], 'centroid_north_sift': 60 - c_mean[1],
            'n_matches_sift': len(masked_matches), 
            }

        return pd.Series(feats)

    def get_features_from_row(self, row):
        base_dir = self.test_dir if pd.isna(row['North']) else self.train_dir
        
        img1, img2 = read_before_and_after(str(base_dir / row.name))

        sift_features = self._get_sift_features(img1, img2)
        lk_features = self._get_lk_features(img1, img2)

        if (sift_features is None) and (lk_features is None):
            return None
        else:
            return pd.concat([lk_features, sift_features])
