import os


for i in range(12):

    command = "srun --partition gpu --mem=40G --gres=gpu:1 -c 4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=" + str(65241+i) + " main_task_retrieval.py --do_eval --num_thread_reader=0 --epochs=1 --batch_size=129 --n_display=5 --train_csv /scratch/shared/beegfs/vlad/clip4clip/msrvtt_data//MSRVTT_train_ex.csv --val_csv /scratch/shared/beegfs/vlad/clip4clip/msrvtt_data/MSRVTT_" +str(i)+"_test.csv --idx_aux "+str(i)+" --data_path /scratch/shared/beegfs/vlad/clip4clip/msrvtt_data//MSRVTT_data.json --features_path /scratch/shared/beegfs/vlad/clip4clip/msrvtt_data//MSRVTT_Videos --lr 1e-4 --output_dir=ckpts/ckpt_msrvtt_retrieval_eval_predict --max_words 32 --max_frames 12 --batch_size_val 129 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0 --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --init_model ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.1  &> log_msrvtt" + str(i)+".txt &"
    print(command )
