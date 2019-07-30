def get_config(alg, dataset):
    config = {'name': 'config'}
    if alg == 'wse':
        if dataset == 'toy':
            config['optimizer'] = 'agd'
            config['embedding_dim'] = 2
            config['type_laplacian'] = 'norm'
            config['speedup'] = 'fast'
            config['epoch'] = 50
            config['graph_partition'] = 2
            config['lambda'] = 1
            config['log_opt.is_log'] = True
            config['log_opt.num_clusters'] = 2
            config['log_opt.disp_steps'] = 5

        elif dataset == 'au_data':
            config['optimizer'] = 'agd'
            config['embedding_dim'] = 10
            config['type_laplacian'] = 'norm'
            config['speedup'] = 'fast'
            config['epoch'] = 50
            config['graph_partition'] = 2
            config['lambda'] = 1
            config['log_opt.is_log'] = False

        elif dataset == 'mnist':
            config['optimizer'] = 'agd'
            config['embedding_dim'] = 2
            config['type_laplacian'] = 'norm'
            config['speedup'] = 'fast'
            config['epoch'] = 50
            config['graph_partition'] = 2
            config['lambda'] = 0.1
            config['log_opt.is_log'] = True
            config['log_opt.num_clusters'] = 2
            config['log_opt.disp_steps'] = 1
    elif alg == 'reannotation':
        if dataset == 'au_data':
            config['batch'] = 1000
            config['num_knn'] = 50
            config['build_params.algorithm'] = 'kdtree'
            config['build_params.trees'] = 1000
            config['build_params.checks'] = 100
            config['remove_outlier'] = False

    return config
