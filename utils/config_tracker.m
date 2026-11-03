
function trackers = config_tracker()
        trackers = {    
                    struct('name', 'DeT',           'publish', 'xxx');
                    struct('name', 'SPT',           'publish', 'xxx'); 
                    struct('name', 'DMTracker',           'publish', 'xxx'); 
                    struct('name', 'HMFT',           'publish', 'xxx'); 
                    struct('name', 'SiamTFA',           'publish', 'xxx'); 
                    struct('name', 'TrDiMP',           'publish', 'xxx'); 
                    struct('name', 'SeqTrack',           'publish', 'xxx'); 
                    struct('name', 'ViPT_RGBD',           'publish', 'xxx'); 
                    struct('name', 'ViPT_RGBT',           'publish', 'xxx'); 
                    struct('name', 'UnTrack_RGBT',           'publish', 'xxx'); 
                    struct('name', 'UnTrack_RGBD',           'publish', 'xxx'); 
                    struct('name', 'SUTrack_RGBT',           'publish', 'xxx');
                    struct('name', 'SUTrack_RGBD',           'publish', 'xxx'); 
                    struct('name', 'MCITrack',           'publish', 'xxx');
                    struct('name', 'SDSTrack_RGBT',           'publish', 'xxx'); 
                    struct('name', 'SDSTrack_RGBD',           'publish', 'xxx'); 
                    struct('name', 'OSTrack',           'publish', 'xxx'); 
                    struct('name', 'ODTrack',           'publish', 'xxx'); 
                    struct('name', 'ATOM',           'publish', 'xxx'); 
                    struct('name', 'DiMP',           'publish', 'xxx'); 
                    struct('name', 'PrDiMP',           'publish', 'xxx'); 
                    struct('name', 'ToMP',           'publish', 'xxx'); 
                    struct('name', 'MixFormer',           'publish', 'xxx'); 
                    struct('name', 'Stark',           'publish', 'xxx'); 
                    struct('name', 'RDTTrack',           'publish', 'xxx'); 
                    struct('name', 'RDTTrack_official',           'publish', 'xxx'); 


                    struct('name', 'OSTrack_official_weight',           'publish', 'xxx');
%                     struct('name', 'OSTrack_train_10',           'publish', 'xxx');
%                     struct('name', 'OSTrack_train_15',           'publish', 'xxx');
%                     struct('name', 'OSTrack_train_25',           'publish', 'xxx');

                    
                    } ; 
%     trackers = {    
% 
% %                     RGBT:
% 
%                     struct('name', 'DeT_04',           'publish', 'xxx'); 
%                     struct('name', 'DeT_15',           'publish', 'xxx'); 
%                     struct('name', 'DeT_35',           'publish', 'xxx'); 
% 
%                     struct('name', 'SPT_15',           'publish', 'xxx'); 
% 
% %                     struct('name', 'DMTracker_03',           'publish', 'xxx'); 
%                     struct('name', 'DMTracker_09',           'publish', 'xxx'); 
% %                     struct('name', 'DMTracker_28',           'publish', 'xxx'); 
% 
%                     struct('name', 'HMFT',           'publish', 'xxx'); 
% 
% %                     struct('name', 'SiamTFA_05',           'publish', 'xxx'); 
% %                     struct('name', 'SiamTFA_70',           'publish', 'xxx'); 
%                     struct('name', 'SiamTFA_10',           'publish', 'xxx'); 
% %                     struct('name', 'SiamTFA_15',           'publish', 'xxx'); 
% %                     struct('name', 'SiamTFA_08',           'publish', 'xxx'); 
% %                     struct('name', 'SiamTFA_11',           'publish', 'xxx'); 
% 
% %                     struct('name', 'TrDiMP_02',           'publish', 'xxx'); 
% %                     struct('name', 'TrDiMP_05',           'publish', 'xxx'); 
%                     struct('name', 'TrDiMP_50',           'publish', 'xxx'); 
% 
%                     struct('name', 'SeqTrack_50',           'publish', 'xxx'); 
% %                     struct('name', 'SeqTrack_30',           'publish', 'xxx'); 
% %                     struct('name', 'SeqTrack_10',           'publish', 'xxx');
% 
% 
% 
% 
% %                     struct('name', 'GT',           'publish', 'xxx'); 
% %                     struct('name', 'ViPT_RGBD_13epoch',           'publish', 'xxx'); 
% %                     struct('name', 'ViPT_RGBD_7epoch',           'publish', 'xxx'); 
%                     struct('name', 'ViPT_RGBD',           'publish', 'xxx'); 
%                     struct('name', 'ViPT_RGBT',           'publish', 'xxx'); 
% 
%                     struct('name', 'UnTrack_RGBT_10',           'publish', 'xxx'); 
% %                     struct('name', 'UnTrack_RGBT_60',           'publish', 'xxx'); 
% 
% %                     struct('name', 'UnTrack_RGBD_10',           'publish', 'xxx'); 
%                     struct('name', 'UnTrack_RGBD_30',           'publish', 'xxx'); 
% %                     struct('name', 'UnTrack_RGBD_60',           'publish', 'xxx');
% 
% %                     struct('name', 'SUTrack_RGBT_pretrained',           'publish', 'xxx'); 
% %                     struct('name', 'SUTrack_RGBT_80',           'publish', 'xxx'); 
%                     struct('name', 'SUTrack_RGBT_60',           'publish', 'xxx'); 
% %                     struct('name', 'SUTrack_RGBT_10',           'publish', 'xxx'); 
% %                     struct('name', 'SUTrack_RGBT_10_30000',           'publish', 'xxx'); 
% %                     struct('name', 'SUTrack_RGBT_20_30000',           'publish', 'xxx'); 
% 
%                     struct('name', 'SUTrack_RGBD_60',           'publish', 'xxx'); 
% %                     struct('name', 'SUTrack_RGBD_10',           'publish', 'xxx'); 
% 
% 
% %                     struct('name', 'MCITrack_50',           'publish', 'xxx');
% %                     struct('name', 'MCITrack_100',           'publish', 'xxx');
% %                     struct('name', 'MCITrack_85_30000',           'publish', 'xxx');
%                     struct('name', 'MCITrack_10_30000',           'publish', 'xxx');
% %                     struct('name', 'MCITrack_5_30000',           'publish', 'xxx');
%                     
% %                     struct('name', 'SDSTrack_RGBT_5',           'publish', 'xxx'); 
%                     struct('name', 'SDSTrack_RGBT_10',           'publish', 'xxx'); 
% %                     struct('name', 'SDSTrack_RGBT_40',           'publish', 'xxx'); 
% %                     struct('name', 'SDSTrack_RGBT_10_30000',           'publish', 'xxx'); 
% 
%                     struct('name', 'SDSTrack_RGBD_10',           'publish', 'xxx'); 
% 
% %                     struct('name', 'STTrack_RGBT',           'publish', 'xxx'); 
% 
% %                     struct('name', 'ViPT_RGBD_ori',           'publish', 'xxx'); 
% %                     struct('name', 'ViPT_RGBT_ori',           'publish', 'xxx'); 
% %                     struct('name', 'ViPT_RGBT_10epoch',           'publish', 'xxx'); 
% %                     struct('name', 'ViPT_RGBT_5epoch',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_13epoch',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_3epoch',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_2epoch',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_1epoch',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_train',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_train_5',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_train_10',           'publish', 'xxx'); 
%                     struct('name', 'OSTrack_train_15',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_train_20',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_train_25',           'publish', 'xxx'); 
% 
% %                     struct('name', 'ODTrack',           'publish', 'xxx'); 
% %                     struct('name', 'ODTrack_train_15',           'publish', 'xxx'); 
%                     struct('name', 'ODTrack_train_10',           'publish', 'xxx'); 
% %                     struct('name', 'ODTrack_train_15',           'publish', 'xxx'); 
% %                     struct('name', 'ODTrack_train_20',           'publish', 'xxx'); 
%                     struct('name', 'ATOM_train',           'publish', 'xxx'); 
% %                     struct('name', 'ATOM',           'publish', 'xxx'); 
%                     struct('name', 'DiMP50_train',           'publish', 'xxx'); 
% %                     struct('name', 'DiMP50',           'publish', 'xxx'); 
% %                     struct('name', 'DiMP18',           'publish', 'xxx'); 
%                     struct('name', 'DiMP18_train',           'publish', 'xxx'); 
% %                     struct('name', 'PrDiMP50',           'publish', 'xxx'); 
%                     struct('name', 'PrDiMP50_train',           'publish', 'xxx'); 
% %                     struct('name', 'PrDiMP18',           'publish', 'xxx'); 
%                     struct('name', 'PrDiMP18_train',           'publish', 'xxx'); 
% %                     struct('name', 'ToMP50',           'publish', 'xxx'); 
%                     struct('name', 'ToMP50_train',           'publish', 'xxx'); 
%                     struct('name', 'ToMP50_train_10',           'publish', 'xxx'); 
% %                     struct('name', 'KeepTrack',           'publish', 'xxx'); 
% 
% %                     struct('name', 'MixFormer_cvt',           'publish', 'xxx'); 
%                     struct('name', 'MixFormer_cvt_train',           'publish', 'xxx'); 
% 
% %                     struct('name', 'MixFormer_vit',           'publish', 'xxx'); 
% %                     struct('name', 'MixFormer_convmae',           'publish', 'xxx'); 
% 
% 
% 
%                     struct('name', 'Stark',           'publish', 'xxx'); 
% %                         struct('name', 'e2e',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'Mamba+ViT_post_add_new_event',           'publish', 'xxx');
% %                     struct('name', 'Mamba+ViT_post_add_old_event',           'publish', 'xxx');
% %                     struct('name', 'Mamba+ViT_pre_add',           'publish', 'xxx');
% %                     struct('name', 'Mamba+ViT_pre_add_old_event',           'publish', 'xxx');
% %                     struct('name', 'Mamba+ViT_pre_add_new_event',           'publish', 'xxx');
% %                     struct('name', 'OSTrack_event_new',           'publish', 'xxx'); 
% %                     struct('name', 'OSTrack_rgb_only',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'OSTrack_rgb',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'OSTrack_event',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'AiATrack',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'ATOM',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'DiMP',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'GRM',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'KYS',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'MixFormer',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'OSTrack',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'PrDiMP',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'STARK',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'SuperDiMP',           'publish', 'xxx');           %% NO. 5 
% %                     struct('name', 'TOMP50',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'TransT',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'AFNet',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'HRCEUTrack',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular12layer',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'Ours',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular8layer',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular6layer',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular12layer+6atten_prompt',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular8layer+6atten_prompt',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular8layer+8atten_prompt',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular8layer+12atten_prompt',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'CEUTrack',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'ceutrack_missing16',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'ceutrack_missing16_40ep',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'ceutrack_prompt16',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'ceutrack_missing12',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'hopular_parallel',           'publish', 'xxx');           %% NO. 5
% %                     struct('name', 'ceutrack_missing16_12layer',           'publish', 'xxx');           %% NO. 5
%                     } ; 
end







