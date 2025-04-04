import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Model Parameters
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset',
                        default='IEMOCAP',
                        help='train/eval dataset')
    parser.add_argument('--use_state_dict',
                        type=bool,
                        default=False,
                        help='use best model state dict to eval')
    parser.add_argument('--state_dict_path',
                        default="../Saved_Models/IEMOCAP/best_model.pth",
                        help='use best model state dict to eval')
    parser.add_argument('--modality',
                        type=str,
                        choices=['t', 'a', 'v', 'ta', 'tv', 'av', 'tav'],
                        default='vat',
                        help='using modality')
    parser.add_argument('--pkl',
                        default='origin',
                        help='dataset pkl: origin/preprocess')
    parser.add_argument('--early_stop_count', type=int, default=5)

    # Dimension Parameters
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=512)
    parser.add_argument('--visual_dim', type=int, default=1000)
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--dual_fusion_layer_num', type=int, default=2)

    # Embedding Parameters
    parser.add_argument('--if_use_speaker_embedding',
                        type=int,
                        default=0,
                        help='using speaker embedding')
    parser.add_argument('--if_use_dialogue_rnn',
                        type=int,
                        default=1,
                        help='using speaker embedding')
    parser.add_argument('--if_use_self_distill',
                        type=int,
                        default=1,
                        help='using self distill')
    parser.add_argument(
        '--if_use_weight_entropy_temp',
        type=float,
        default=1.0,
        help='using weight entropy to control weight softmax temperature')
    parser.add_argument('--speaker_embedding_ratio',
                        type=float,
                        default=1,
                        help='1')
    parser.add_argument('--if_shared_speaker_embedding',
                        type=bool,
                        default=False,
                        help='using shared speaker embedding')

    # Self Distill Parameters
    parser.add_argument('--self_distill_num',
                        type=int,
                        default=0,
                        help='features num use self distill')

    # Train Parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='IEMOCAP 100 MELD 100')
    parser.add_argument('--mid_epoch_num',
                        type=int,
                        default=10,
                        help='when to change training focus')
    parser.add_argument('--late_epoch_num',
                        type=int,
                        default=20,
                        help='when to change training focus')
    parser.add_argument('--num_heads', type=int, default=8, help='8')
    parser.add_argument('--gradient_accumulate_step',
                        type=int,
                        default=8,
                        help='gradient accumulate step')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='0.1')
    parser.add_argument('--lr_main_task',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_gen',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_adv',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0001,
                        help='opt.Adm weight decay')
    parser.add_argument('--schd_fc',
                        type=float,
                        default=0.7,
                        help='scheduler factor')
    parser.add_argument('--schd_patience',
                        type=int,
                        default=3,
                        help='scheduler wait patience')
    parser.add_argument('--schd_adv_gen_step',
                        type=int,
                        default=3,
                        help='scheduler adv gen wait patience')
    parser.add_argument('--split_valid',
                        type=float,
                        default=0.1,
                        help='ratio to split train and validation')

    # Loss Parameters
    parser.add_argument('--single_loss_param', type=float, default=0.1)
    parser.add_argument('--adv_loss_param', type=float, default=0.1)
    parser.add_argument('--recon_loss_param', type=float, default=1.0)
    parser.add_argument('--single_CC_recon_loss_param',
                        type=float,
                        default=1.0)
    parser.add_argument('--mm_CS_recon_loss_param', type=float, default=1.0)
    parser.add_argument('--dual_loss_param', type=float, default=0.3)
    parser.add_argument('--teacher_CE_loss_param', type=float, default=1.0)
    parser.add_argument('--student_MSE_loss_param', type=float, default=1.0)
    parser.add_argument('--weight_reg_loss_param', type=float, default=1.0)
    parser.add_argument('--MMG_KL_loss_param', type=float, default=1.0)
    parser.add_argument('--DAF_loss_param', type=float, default=0.4)
    # parser.add_argument('--1', type=float, default=1, help='1')
    return parser.parse_args()
