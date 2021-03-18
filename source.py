import cv2
import numpy as np
import os
import random


def getInputData():
    path = os.path.dirname(os.path.realpath(__file__))
    file_names = [filename for filename in os.listdir(path+'\\Assignment_MV_02_calibration\\') if filename.endswith("png")]
    chessImages = [os.path.join(path+'\\Assignment_MV_02_calibration\\', fn) for fn in file_names]
    vid_name = "Assignment_MV_02_video.mp4"
    video = os.path.join(path, vid_name)
    return chessImages, video

def saveImg(fname, img):
    path = os.path.dirname(os.path.realpath(__file__))
    cv2.imwrite(os.path.join(path+'\\res\\' , fname), img)

def chessBoardCalibrate(full_path):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((5*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in full_path:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7,5), None)

        if ret == True:
            objpoints.append(objp)
            # get subpixel accurate corner points
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,5), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            saveImg((fname.split("\\"))[-1], img)

    ret, mtx_K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    from matplotlib import pyplot as plt
    plt.imshow(mtx_K)
    plt.show()
    return mtx_K

def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame=0
    while camera.isOpened():
        ret,img= camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame>last_frame:
            break
    return result

def firstFrameFeatures(video):
    # extract 1st frame
    images = extract_frames(video, [1])
    img = images[1].copy()
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # obtain features from extracted frame
    corners = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # get features with subpixel accuracy
    p0 = cv2.cornerSubPix(new_img,np.float32(corners),(5,5),(-1,-1),criteria)
    # visualize feature points
    for i,points in enumerate(p0):
        x,y = points[0]
        cv2.circle(img,(x,y), 5, (0,255,0), -1)
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    saveImg("firstFrameFeatures.png", img)
    return img


def featureTracking(video):
    stream = cv2.VideoCapture(video)

    while stream.isOpened():
        ret,img= stream.read()        
        if ret:
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        
            p0 = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)    
            # refine features to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            p0 = cv2.cornerSubPix(new_img,np.float32(p0),(5,5),(-1,-1),criteria)                                
            break  
    
    # initialise tracks
    index = np.arange(len(p0))
    tracks = {}
    for i in range(len(p0)):
        tracks[index[i]] = {0:p0[i]}
                
    frame = 0
    frame_img=[]
    while stream.isOpened():
        ret,img= stream.read()                 
        if not ret:
            break

        frame += 1

        old_img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        

        # calculate optical flow
        if len(p0)>0: 
            p1, st, err  = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None)                                    
            
            # visualise points
            for i in range(len(st)):
                if st[i]:
                    cv2.circle(img, (p1[i,0,0],p1[i,0,1]), 2, (0,0,255), 2)
                    cv2.line(img, (p0[i,0,0],p0[i,0,1]), (int(p0[i,0,0]+(p1[i][0,0]-p0[i,0,0])*5),int(p0[i,0,1]+(p1[i][0,1]-p0[i,0,1])*5)), (0,0,255), 2)            
            
            p0 = p1[st==1].reshape(-1,1,2)            
            index = index[st.flatten()==1]
            
        # refresh features, if too many lost
        if len(p0)<100:
            new_p0 = cv2.goodFeaturesToTrack(new_img, 200-len(p0), 0.3, 7)
            # refine to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            new_p0 = cv2.cornerSubPix(new_img,np.float32(new_p0),(5,5),(-1,-1),criteria)  
            for i in range(len(new_p0)):
                if np.min(np.linalg.norm((p0 - new_p0[i]).reshape(len(p0),2),axis=1))>10:
                    p0 = np.append(p0,new_p0[i].reshape(-1,1,2),axis=0)
                    index = np.append(index,np.max(index)+1)

        # update tracks
        for i in range(len(p0)):
            if index[i] in tracks:
                tracks[index[i]][frame] = p0[i]
            else:
                tracks[index[i]] = {frame: p0[i]}

        # visualise last frames of active tracks
        for i in range(len(index)):
            for f in range(frame-20,frame):
                if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                    cv2.line(img,
                            (tracks[index[i]][f][0,0],tracks[index[i]][f][0,1]),
                            (tracks[index[i]][f+1][0,0],tracks[index[i]][f+1][0,1]), 
                            (0,255,0), 1)
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        
        frame_img.append(img)
        cv2.imshow("camera", img)    

    cv2.imshow("last frame", frame_img[-1])
    cv2.waitKey()
    cv2.destroyAllWindows()
    saveImg("tracks.png", frame_img[-1])

    stream.release()
    return tracks, frame

def getCorrespondences(tracks, frame1, frame2):
    correspondences = []
    # loop through tracks to get frame correspondences
    for track in tracks:
        if (frame1 in tracks[track]) and (frame2 in tracks[track]):
            x1 = [tracks[track][frame1][0,0],tracks[track][frame1][0,1],1]
            x2 = [tracks[track][frame2][0,0],tracks[track][frame2][0,1],1]
            correspondences.append((np.array(x1), np.array(x2)))
    return correspondences

def getTMatrices(correspondences):
    # calculate mean and standard deviation for each correspondence
    mean = np.squeeze(np.mean(np.array([correspondences]), axis=1))
    std = np.squeeze(np.std(np.array([correspondences]), axis=1))

    # get T and T_
    T = np.array([[1/std[0][0], 0, -mean[0][0]/std[0][0]],
                    [0,1/std[0][1], -mean[0][1]/std[0][1]],
                    [0,0,1]])

    T_ = np.array([[1/std[1][0], 0, -mean[1][0]/std[1][0]],
                    [0,1/std[1][1], -mean[1][1]/std[1][1]],
                    [0,0,1]])

    return correspondences, T, T_

def getFundamentalMatrix(correspondences, T, T_, video):
    # random.seed(123)
    best_outliers = len(correspondences) + 1
    best_error = 1e100
    best_H = np.eye(3)

    # get first and last frame
    images = extract_frames(video,[1,30])
    img = images[1].copy()
    last_img = images[30].copy()
    # inliers = []
    for i in range(10000):
        # obtain 8 random correspondences
        random_idx = random.sample(range(len(correspondences)), 8)
        random_correspondences = [correspondences[idx] for idx in random_idx]

        # normalize the 8 random correspondences
        # build matrix using said correspondences
        A = np.zeros((0,9))
        for x1,x2 in random_correspondences:
            x1n = np.matmul(T,x1)
            x2n = np.matmul(T_,x2) 
            ai = np.kron(x1n.T,x2n)
            A = np.append(A,[ai],axis=0)
                
        U,S,V = np.linalg.svd(A)
        F = V[8,:].reshape(3,3).T

        U,S,V = np.linalg.svd(F)
        # calculate fundamental matrix
        F = np.matmul(U,np.matmul(np.diag([S[0],S[1],0]),V))
        F = np.matmul(T_.T, np.matmul(F, T))

        # get unselected correspondences
        new_correspondences = [element for i, element in enumerate(correspondences) if i not in random_idx]
        x = np.squeeze(np.array([new_correspondences]))[:,0]
        x_ = np.squeeze(np.array([new_correspondences]))[:,1]

        # point observation covariance matrix
        Cxx = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,0]])
        
        count_outliers = 0
        accumulate_error = 0
        inliers, outliers = [], []
        # loop through unselected correspondences
        for x1,x2 in new_correspondences:
            # calculate gi and variance
            value = np.matmul(np.matmul(x2.T, F), x1)

            variance = np.matmul(np.matmul(np.matmul(np.matmul(x2.T, F), Cxx), F.T), x2)
            variance += np.matmul(np.matmul(np.matmul(np.matmul(x1.T, F.T), Cxx), F), x1)
            
            # get test statistics using gi and variance
            test_stat = np.square(value) / variance

            # threshold the test statistics
            if test_stat > 6.635:
                # count outliers
                count_outliers += 1
                # record outliers
                outliers.append((x1,x2))
            else:
                # sum the test statistics for inliers
                accumulate_error += test_stat
                # record inliers
                inliers.append((x1,x2))

        # to record lowest amount of outliers
        if count_outliers < best_outliers:
            best_error = accumulate_error
            best_outliers = count_outliers
            # get fundamental matrix and inliers for lowest amount of outliers
            final_Fmat = F
            final_inliers = inliers
            final_outliers = outliers
        # break ties for number of outliers
        elif count_outliers ==  best_outliers:
            # compare sum of test statistics when tied for number of outliers
            if accumulate_error < best_error:
                best_error = accumulate_error
                best_outliers = count_outliers
                # get fundamental matrix and inliers
                final_Fmat = F
                final_inliers = inliers
                final_outliers = outliers

    # draw inliers (blue) and outliers (red) on first and last frames
    for x1,x2 in final_inliers:
        cv2.circle(img, (int(x1[0]), int(x1[1])), 5, (0, 0, 255), -1)
        cv2.circle(last_img, (int(x2[0]), int(x2[1])), 5, (0, 0, 255), -1)
    
    for x1,x2 in final_outliers:
        cv2.circle(img, (int(x1[0]),int(x1[1])), 5, (255, 0, 0), -1)
        cv2.circle(last_img, (int(x2[0]), int(x2[1])), 5, (255, 0, 0), -1)

    cv2.imshow("first frame", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    saveImg("in_out_firstFrame.png",img)

    cv2.imshow("last frame", last_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    saveImg("in_out_lastFrame.png",last_img)
    return final_Fmat, final_inliers

def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]

    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]

    return e1,e2 

def calculateEssentialMatrix(matrix_K, F):
    # calculate essential mat
    E = np.matmul(np.matmul(matrix_K.T, F), matrix_K)

    # decompose E
    U,S,V = np.linalg.svd(E)

    # ensure positive determinants
    if np.linalg.det(U)<0:
        U[:,2] *= -1
    if np.linalg.det(V)<0:
        V[:,2] *= -1
    return U,V

def getRotate_Translate(U,V):
    W = np.array([[0,-1,0],
                [1,0,0],
                [0,0,1]])

    Z = np.array([[0,1,0],
                [-1,0,0],
                [0,0,0]])

    # assume 50km/h and 30fps
    B = (50000/3600) * (30/30)

    # skew of R.T and t
    skew_1 = B * np.matmul(U,np.matmul(Z,U.T))
    skew_2 = -(skew_1)

    # get R.T * t
    X_POS = [skew_1[2,1], skew_1[0,2], skew_1[1,0]]
    X_NEG = [skew_2[2,1], skew_2[0,2], skew_2[1,0]]

    # R1,R2
    R1 = np.matmul(U, np.matmul(W, V.T)) 
    R2 = np.matmul(U, np.matmul(W.T, V.T))

    # 4 translations matrices
    t1= np.matmul(np.linalg.inv(R1), X_POS)
    t2= np.matmul(np.linalg.inv(R1), X_NEG)
    t3= np.matmul(np.linalg.inv(R2), X_POS)
    t4= np.matmul(np.linalg.inv(R2), X_NEG)

    # 4 possible combinations
    combo = [(R1,t1), (R1,t2), (R2,t2), (R2,t2)]

    return combo

def plot3d(x,y,z, t):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plot 3d points
    ax.scatter3D(x, y, z, c='r')

    # plot 1st and 2nd camera center
    ax.scatter3D(0,0,0, c='b')
    ax.scatter3D(t[0],t[1],t[2], c='b')

    plt.show()

def get3dpoints(combo, inliers, matrix_K):
    print ("length of inliers:",len(inliers))

    # 3d point coords
    x,y,z = [],[],[]
    best_solution = len(inliers)+1
    # for each translation and rotation combination
    for c in combo:
        r,t = c[0], c[1]

        count = 0
        # loop through inliers
        for x1,x2 in inliers:
            m = np.matmul(np.linalg.inv(matrix_K),x1)
            m_ = np.matmul(np.linalg.inv(matrix_K),x2)

            # linear equation set up
            LHS = np.array([[np.matmul(m.T, m), np.matmul(-m.T, np.matmul(r.T, m_))],
                            [np.matmul(m.T, np.matmul(r.T, m_)), np.matmul(-m.T, m_)]])
            RHS = np.array([np.matmul(t.T, m), 
                            np.matmul(t.T, np.matmul(r.T, m_))])

            # solve linear equation
            # returns 2 element array containing results from linear equation
            res = np.matmul(np.linalg.inv(LHS), RHS)
            lambd, mue = res[0], res[1]
            
            # if points are in front of both cameras
            if lambd > 0 and mue > 0:
                count+=1
                # calculate the 3d points for both frame correspondence
                x_lambd = lambd * m
                x_mue = t + (mue*np.matmul(r.T, m_))

                # obtain final 3d point by averaging the distance between xlambda and xmue
                x_final = (x_lambd+x_mue) / 2
                x.append(x_final[0])
                y.append(x_final[1])
                z.append(x_final[2])
        
        # get 3d coordinates and translation vector that yielded the best solution
        if count < best_solution:
            best_solution = count
            best_x = x
            best_y = y
            best_z = z
            best_t = t
                
    print ("number of 3d points",len(z))
    return best_x, best_y, best_z, best_t


def main():
    #==========
    #  TASK 1
    #==========
    # subtask A, B
    chessImages, video = getInputData()
    matrix_K = chessBoardCalibrate(chessImages)
    # subtask C
    img = firstFrameFeatures(video)
    # subtask D
    tracks, frame = featureTracking(video)

    #==========
    #  TASK 2
    #==========
    # subtask A
    frame1, frame2 = 1, frame
    correspondences = getCorrespondences(tracks, frame1, frame2)
    # subtask B
    correspondences, T, T_ = getTMatrices(correspondences)
    # subtask C,D,E,F,G
    F, inliers = getFundamentalMatrix(correspondences, T, T_, video)
    # subtask H
    e1,e2 = calculate_epipoles(F)

    #==========
    #  TASK 3
    #==========
    # subtask A
    U,V = calculateEssentialMatrix(matrix_K, F)
    # subtask B
    RT_combination = getRotate_Translate(U,V)
    # subtask C
    x,y,z, t = get3dpoints(RT_combination, inliers, matrix_K)
    # subtask D
    plot3d(x,y,z, t)

main()
