# Dataset specification

```shell
cat scannetv2_train.txt | xargs -n1 -I{} mv scans/{} ../deploy/3dDenseCap/data/scannetv2/train

cd ../deploy/3dDenseCap/data/scannetv2/train

mv */* .
rm -r */
```
Preprocessing:

ScanNet Data:

- \[scene_id\]_vh_clean_2.ply
  - Cleaned and decimated mesh for semantic annotations
- \[scene_id\]_vh_clean_2.labels.ply
  - Visualization of aggregated semantic segmentation; colored by nyu40 labels (see img/legend; ply property 'label' denotes the nyu40 label id)
- \[scene_id\]_vh_clean_2.0.010000.segs.json
  - Over-segmentation of annotation mesh
- \[scene_id\].aggregation.json
  - Aggregated instance-level semantic annotations on lo-res meshes
- scannetv2-labels.combined.tsv
  - General semantic label definitions

[//]: # (prepare_data_inst.py, outputs:)

[//]: # ()
[//]: # (- \[scene_id\]_inst_nostuff.pth )

[//]: # (  - coords: N x 3 )

[//]: # (  - colors: N x 3 )

[//]: # (  - sem_labels: N )

[//]: # (  - instance_labels: N)

[//]: # ()
[//]: # (&#40;N = number of points in the scene&#41;)

From Scan2Cap, batch_load_scannet_data.py, outputs:

- '_vert.npy'
  - mesh_vertices (== points): N x 9 (x,y,z, r,g,b, normals nx,ny,nz)
  - **Unlike in pointgroup, here the coordinates are not normalized (mean and std)**
- '_aligned_vert.npy'
  - aligned_vertices: N x 9
- '_sem_label.npy'
  - label_ids: N
- '_ins_label.npy'
  - instance_ids: N
- '_bbox.npy', 
  - instance_bboxes: num_objects x 8: (cx,cy,cz, dx,dy,dz, label id, object id)
    - where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
- '_aligned_bbox.npy', 
  - aligned_instance_bboxes