# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, StateTransitionDataset, TransitionDataset, StoryboardDataset, LEGODataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'state_transition': StateTransitionDataset,
    'storyboard': StoryboardDataset,
    'lego': LEGODataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '<your_path_to_bagel_example>/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '<your_path_to_bagel_example>/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": '<your_path_to_bagel_example>/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '<your_path_to_bagel_example>/bagel_example/vlm/images',
			'jsonl_path': '<your_path_to_bagel_example>/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'state_transition':{
        'key_frames_10k_videos': {
            'data_dir': '<your_path_to_state_transition_data>',
            'jsonl_path': '<your_path_to_state_transition_data>/state_transition_mwd6_ol3.jsonl',
            'num_total_samples': 48260,
		},
    },
    'storyboard':{
        'vinabench_vwp': {
            'data_dir': '<your_path_to_vinabench_vwp_data>',
            'jsonl_path': '<your_path_to_vinabench_vwp_data>/vwp_train_fwd_vwm.jsonl',
            'num_total_samples': 11652,
		},
    },
    'lego':{
        'lego_train': {
            'data_dir': '<your_path_to_lego_data>',
            'jsonl_path': '<your_path_to_lego_data>/lego_train_fwd_vwm.jsonl',
            'num_total_samples': 147362,
		},
    },
}
