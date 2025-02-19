python demo.py  -s data/vgp/dataset/caterpillar --model_path outputs/caterpillar  --white_background --newscaling   --disable_viewer  --eval  \
        --colmap_scale 0.25 --num_segments_per_bprimitive_edge 3  --grad_threshold 0.0018 --edge_threshold 13 --vis_threshold 0.1




python demo.py  -s data/vgp/dataset/barn --model_path outputs/barn  --white_background --newscaling   --disable_viewer  --eval  \
        --colmap_scale 0.1 --num_segments_per_bprimitive_edge 3  --grad_threshold 0.0018 --edge_threshold 13 --vis_threshold 0.1

python demo.py  -s data/vgp/dataset/family --model_path outputs/family  --white_background --newscaling   --disable_viewer  --eval  \
        --colmap_scale 0.1 --num_segments_per_bprimitive_edge 3  --grad_threshold 0.0018 --edge_threshold 13 --vis_threshold 0.1


python demo.py  -s data/vgp/dataset/ignatius --model_path outputs/ignatius   --newscaling   --disable_viewer  --eval -\
        --colmap_scale 0.1 --num_segments_per_bprimitive_edge 3  --grad_threshold 0.0018 --edge_threshold 13 --vis_threshold 0.1


python demo.py  -s data/vgp/dataset/truck --model_path outputs/truck  --white_background --newscaling   --disable_viewer  --eval \
        --colmap_scale 0.35 --num_segments_per_bprimitive_edge 3  --grad_threshold 0.0018 --edge_threshold 13 --vis_threshold 0.1

