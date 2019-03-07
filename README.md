## Project: 3D Perception

This write-up describe the process of 3D perception pipeline - objects recognition using RGBD camera. The process including voxel grid down-sampling, filtering, RANSAC segmentation, clustering, features extraction, SVM training and finally object recognition.

[tabletop]: ./misc_images/tabletop.JPG
[voxelds]: ./misc_images/voxel_downsampled.JPG
[passthru]: ./misc_images/pass_through_filtered.JPG
[inlier]: ./misc_images/extracted_inliers.JPG
[outlier]: ./misc_images/extracted_outliers.JPG
[segcluster]: ./misc_images/seg_cluster.JPG
[svmtrain]: ./misc_images/SVM_train.JPG
[recognition]: ./misc_images/recognition.JPG
[w1]: ./misc_images/world1.JPG
[w2]: ./misc_images/world2.JPG
[w3]: ./misc_images/world3.JPG



### A. Perception pipeline implemented

This section introduces the step-by-step perception pipeline implemented



#### Accessing the RGBD Camera Data

The first step is to read the data coming from the RGBD camera by creating a ROS node `perception` and point cloud subscriber `pcl_sub`. The `pcl_sub` is subscribing to `/pr2/world/points` topic:

```python
# ROS node initialization
rospy.init_node('perception', anonymous=True)
# Create Subscribers
pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
```

The `pcl_callback` function will be called when the RGBD camera publishes a new `pc2.PointCloud2` message.



#### Point Cloud Filtering

Raw data from RGBD camera provide too much information, or say too detailed information. It is often desirable to work with lower resolution, and filter unwanted area but keep specific region of interest.

To reduce the resolution, Voxel Grid filter is used to downsample the input point cloud:

```python
# Convert ROS msg to PCL data
pcl_data = ros_to_pcl(pcl_msg)

# Voxel Grid Downsampling
vox = pcl_data.make_voxel_grid_filter()
LEAF_SIZE = 0.01   
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud_filtered = vox.filter()
```

** Images from Exercise 1 are used to demonstrate the idea of filter

| Raw point cloud | Downsampled Point cloud |
| :-------------: | :---------------------: |
|  ![tabletop][]  |      ![voxelds][]       |

In the images above, the point cloud actually includes unwanted parts: table and floor. Thus, applying pass through filter to keep everything on top of table:

```python
# TODO: PassThrough Filter
passthrough = cloud_filtered.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.7
axis_max = 1.1
passthrough.set_filter_limits(axis_min, axis_max)
cloud_filtered = passthrough.filter()

passthrough2 = cloud_filtered.make_passthrough_filter()
filter_axis = 'y'
passthrough2.set_filter_field_name(filter_axis)
axis_min = -0.5
axis_max = +0.5
passthrough2.set_filter_limits(axis_min, axis_max)
cloud_filtered = passthrough2.filter()
```

** Images from Exercise 1 are used to demonstrate the idea of filter

| Before Pass Through Filter | After Pass Through Filter |
| :------------------------: | :-----------------------: |
|        ![voxelds][]        |       ![passthru][]       |



#### RANSAC Plane Segmentation 

Now that surface of table and everything on top of table are kept, but the surface of table is still unwanted. The surface of table can be separate using RANSAC plane segmentation:

```python
# RANSAC Plane Segmentation
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.01
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()  

# Extract inliers and outliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
```

** Images from Exercise 1 are used to demonstrate the idea of RANSAC

| Original Point Cloud | Extracted Inlier (Table) | Extracted Outlier (Objects) |
| -------------------- | ------------------------ | --------------------------- |
| ![passthru][]        | ![inlier][]              | ![outlier][]                |



#### Clustering - Euclidean Clustering 

Computer doesn't see thing in the same way like human does. Thus, Euclidean Clustering is used to distinguish the objects from one another. This approach differs from k-means in the sense that it doesn't require the prior knowledge of the number of objects we are trying to detect. Unfortunately it uses an hierarchical representation of the point cloud that can be computationally expensive to obtain.

```python
# Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(extracted_outliers)
tree = white_cloud.make_kdtree()

ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.02)
ec.set_MinClusterSize(10)
ec.set_MaxClusterSize(3000)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()

# Create Cluster-Mask Point Cloud to visualize each cluster separately
cluster_color = get_color_list(len(cluster_indices))

color_cluster_point_list = []

for j, indices in enumerate(cluster_indices):
	for i, indice in enumerate(indices):
			color_cluster_point_list.append([white_cloud[indice][0], white_cloud[indice][1], white_cloud[indice][2], rgb_to_float(cluster_color[j])])

cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)
```

** Image of Exercise 2 is used to demonstrate the idea of clustering

| Clustering      |
| --------------- |
| ![segcluster][] |



#### Extracting Point Cloud Features

After distinguishing objects, now it's time for computer to recognize the objects, using Machine Learning tool: Support Vector Machine (SVM). SVMs are not as sophisticated as Deep Neural Nets and they require some hand crafted features to work properly. The features that I’ve created are color and normals histograms concatenated together: 

```python
# Extract histogram features
chists = compute_color_histograms(ros_cluster, using_hsv=True)
normals = get_normals(ros_cluster)
nhists = compute_normal_histograms(normals)
feature = np.concatenate((chists, nhists))
```

Histogram used to capture the overall color (HSV) and shape characteristics of point clouds but with a limited number of dimensions.



#### SVM Model Training

After using `capture_features.py` to get the labeled dataset, train the SVM Model using `train_svm.py`. The kernel type of SVC is `linear` in this case. Here show the training result, where `200` dataset are collected for each objects, and get accuracy of around 97%:

| SVM training  |
| ------------- |
| ![svmtrain][] |



#### Object Recognition and Results

Once the SVM trained, `model_3.sav` is generated as a model reference for objects recognition. 

```python
# Make the prediction, retrieve the label for the result
# and add it to detected_objects_labels list
prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
label = encoder.inverse_transform(prediction)[0]
detected_objects_labels.append(label)

# Publish a label into RViz
label_pos = list(white_cloud[pts_list[0]])
label_pos[2] += .4
object_markers_pub.publish(make_label(label,label_pos, index))
```



Following are the results for all table top configurations (Worlds 1, 2 and 3): 

| World # | Recognition screenshot | Result |
| :-----: | :--------------------: | :----: |
|    1    |        ![w1][]         |  3/3   |
|    2    |        ![w2][]         |  5/5   |
|    3    |        ![w3][]         |  8/8   |



### B. Pick & Place 

This section introduces the final piece of project -  calculate all necessary arguments to call the **pick_place_routine** service to perform a successful pick and place operation, and output to a yaml file.



#### Reading Parameters

The object list and dropbox locations where retrieved as a list from the parameter server. Then converted them into dictionaries making it easier to lookup the configuration values for a specific object:

```python
# Get/Read parameters
object_list_param = rospy.get_param('/object_list')
dropbox_list_param = rospy.get_param('/dropbox')

# Parse parameters into individual variables
object_param_dict = {}
dropbox_param_dict = {}
for i in range(0, len(object_list_param)):
	object_param_dict[object_list_param[i]['name']] = object_list_param[i]

for i in range(0, len(dropbox_list_param)):
	dropbox_param_dict[dropbox_list_param[i]['group']] = dropbox_list_param[i]
```



#### Calculating Objects' Centroid & Pose

Loop through the list of detected objects, and calculate its centroid by averaging all points. Which arm of PR2 to be used and which dropbox to be placed are indicated in dictionaries. Then, create a list of dictionaries by calling `make_yaml_dict` for later output to yaml:

```python
# Loop through the pick list
for object in object_list:

	# Get the PointCloud for a given object and obtain it's centroid
	points_arr = ros_to_pcl(object.cloud).to_array()
	centroid = np.mean(points_arr, axis=0)[:3]

	object_param = object_param_dict[object.label]
	dropbox_param = dropbox_param_dict[object_param['group']]
	object_name.data = str(object.label)

	pick_pose.position.x = np.asscalar(centroid[0])
	pick_pose.position.y = np.asscalar(centroid[1])
	pick_pose.position.z = np.asscalar(centroid[2])
	pick_pose.orientation.x = 0.0
	pick_pose.orientation.y = 0.0
	pick_pose.orientation.z = 0.0
	pick_pose.orientation.w = 0.0

	# Create 'place_pose' for the object
	place_pose.position.x = float(position[0])
	place_pose.position.y = float(position[1])
	place_pose.position.z = float(position[2])
	place_pose.orientation.x = 0.0
	place_pose.orientation.y = 0.0
	place_pose.orientation.z = 0.0
	place_pose.orientation.w = 0.0

	# Assign the arm to be used for pick_place
	arm_name.data = str(dropbox_param['name'])
	
	# Create a list of dictionaries for later output to yaml format
	yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
	dict_list.append(yaml_dict)
```



#### Creating the .yaml Output Files

Once all parameters are set in `dict_list  `, convert and output `dict_list` to yaml format by calling `send_to_yaml()`. 

```python
# Output your request parameters into output yaml file
send_to_yaml("output_"+str(WORLD_ID)+".yaml", dict_list)
```



### Conclusion

This was a fun project and gave me a really good overview of point clouds, how to create a perception pipeline to identify objects and how to integrate perception and motion planning modules using ROS.