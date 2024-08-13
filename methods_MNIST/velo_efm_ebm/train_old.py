import argparse
import mlp
import torch
import numpy as np
# import samplers
# import block_samplers
import torch.nn as nn
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import vamp_utils
import utils.ais as ais
import copy
import time
from velo_mask_efm_ebm.model import EBM


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# def get_sampler(args):
#     data_dim = np.prod(args.input_size)
#     if args.input_type == "binary":
#         if args.sampler == "gibbs":
#             sampler = samplers.PerDimGibbsSampler(data_dim, rand=False)
#         elif args.sampler == "rand_gibbs":
#             sampler = samplers.PerDimGibbsSampler(data_dim, rand=True)
#         elif args.sampler.startswith("bg-"):
#             block_size = int(args.sampler.split('-')[1])
#             sampler = block_samplers.BlockGibbsSampler(data_dim, block_size)
#         elif args.sampler.startswith("hb-"):
#             block_size, hamming_dist = [int(v) for v in args.sampler.split('-')[1:]]
#             sampler = block_samplers.HammingBallSampler(data_dim, block_size, hamming_dist)
#         elif args.sampler == "gwg":
#             sampler = samplers.DiffSampler(data_dim, 1,
#                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
#         elif args.sampler.startswith("gwg-"):
#             n_hops = int(args.sampler.split('-')[1])
#             sampler = samplers.MultiDiffSampler(data_dim, 1, approx=True, temp=2., n_samples=n_hops)
#         elif args.sampler == "dmala":
#             sampler = samplers.LangevinSampler(data_dim, 1,
#                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=True)

#         elif args.sampler == "dula":
#             sampler = samplers.LangevinSampler(data_dim, 1,
#                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=False)

        
#         else:
#             raise ValueError("Invalid sampler...")
#     else:
#         if args.sampler == "gibbs":
#             sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=False)
#         elif args.sampler == "rand_gibbs":
#             sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=True)
#         elif args.sampler == "gwg":
#             sampler = samplers.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
#         else:
#             raise ValueError("invalid sampler")
#     return sampler

def gen_samples(model, args, batch_size=None, t=0.0, xt=None):
    model.eval()
    S, D = args.vocab_size_with_mask, args.discrete_dim

    # Variables, B, D for batch size and number of dimensions respectively
    B = batch_size if batch_size is not None else args.batch_size

    M = S - 1

    # Initialize xt with the mask index value if not provided
    if xt is None:
        xt = M * torch.ones((B, D), dtype=torch.long).to(args.device)


    dt = args.delta_t  # Time step
    t = 0.0  # Initial time

    while t < 1.0:
        t_ = t * torch.ones((B,)).to(args.device)
        with torch.no_grad():
            x1_logits = model(xt, t_).to(args.device)
            x1_logits = F.softmax(x1_logits, dim=-1)
        delta_xt = torch.zeros((B,D,S)).to(args.device)
        delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0) 
        ut = 1/(1-t) * (x1_logits - delta_xt)

        step_probs = delta_xt + (ut * dt)

        if args.impute_self_connections:
            step_probs = step_probs.clamp(max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 

        t += dt

        step_probs = step_probs.clamp(min=0) #can I avoid this?


        if t < 1.0:
            xt = Categorical(step_probs).sample() #(B,D)
        else:
            print(f'final t at {t}')
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Number of masked entries in the final but one tensor: {num_masked_entries}")
                print(f"Forcing mask values into range...")
            print(f'Share of samples with non-zero probability for at least one mask: {(step_probs[:,:,M].sum(dim=-1)>0.001).sum()/B}')
            step_probs[:, :, M] = 0
            step_probs_sum = step_probs.sum(dim=-1, keepdim=True)
            zero_sum_mask = step_probs_sum == 0
            if zero_sum_mask.any():
                step_probs[zero_sum_mask.expand(-1, -1, S).bool() & (torch.arange(S).to(args.device) < M).unsqueeze(0).unsqueeze(0).expand(B, D, S)] = 1/M
            # print(step_probs[zero_sum_mask.expand(-1, -1, S)])
            xt = Categorical(step_probs).sample() # (B, D)
            if torch.any(xt == M):
                num_masked_entries = torch.sum(xt == M).item()
                print(f"Forcing failed. Number of masked entries in the final tensor: {num_masked_entries}")

    return xt.detach().cpu().numpy()

def compute_loss(ebm_model, temp, dfs_model, q_dist, xt, x1, t, args):
    (B, D), S = x1.size(), args.vocab_size_with_mask
    M = S - 1

    x1_logits = dfs_model(xt, t).to(args.device)

    loss = F.cross_entropy(x1_logits.transpose(1,2), x1, reduction='none').sum(dim=-1) #maybe need to ignore index
    # if args.impute_self_connections:            #is this appropriate here?
    #     loss.scatter_(-1, xt[:, :, None], 0.0)
    #loss = loss.sum(dim=(1, 2))

    x_hat = torch.argmax(x1_logits, dim=-1)
    acc = (x_hat == x1).float().mean().item()

    log_prob = -ebm_model(x1.float()) / temp
    print(f'max is {log_prob.max().item()}')
    print(f'min is {log_prob.min().item()}')
    log_q_density = q_dist.get_last_log_likelihood()
    log_weights = log_prob - log_q_density
    if args.norm_by_sum and not args.norm_by_max:
        log_norm = torch.logsumexp(log_weights, dim=-1) #make this a moving average?
    elif args.norm_by_max and not args.norm_by_sum:
        log_norm = torch.max(log_weights)
    else:
        raise NotImplementedError('Must either normalize by sum or max to avoid inf.')
    # print(f'\n log norm is {log_norm} \n')
    weights = (log_weights - log_norm).exp()
    loss = weights * loss #the math is wrong?
    loss = loss.mean(dim=0)

    return loss, acc, weights



def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')
        logger.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))
    
    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    # make dfs model
    (B, D), S = x1.size(), args.vocab_size_with_mask
    M = S - 1
    dfs_model = ResNetFlow(64, args)
    
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = (init_batch.mean(0) * (1. - 2 * eps) + eps).to(args.device)
    q_dist = Dataq(args, ema_dfs_model, bernoulli_mean=init_mean)

    # make model
    if args.ebm_model.startswith("mlp-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.mlp_ebm(np.prod(args.input_size), nint)
    elif args.ebm_model.startswith("resnet-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    elif args.ebm_model.startswith("cnn-"):
        nint = int(args.ebm_model.split('-')[1])
        net = mlp.MNISTConvNet(nint)
    else:
        raise ValueError("invalid ebm_model definition")

    # get data mean and initialize buffer
    # init_batch = []
    # for x, _ in train_loader:
    #     init_batch.append(preprocess(x))
    # init_batch = torch.cat(init_batch, 0)
    # eps = 1e-2
    # init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps
    # if args.buffer_init == "mean":
    #     if args.input_type == "binary":
    #         init_dist = torch.distributions.Bernoulli(probs=init_mean)
    #         buffer = init_dist.sample((args.buffer_size,))
    #     else:
    #         buffer = None
    #         raise ValueError("Other types of data not yet implemented")

    # elif args.buffer_init == "data":
    #     all_inds = list(range(init_batch.size(0)))
    #     init_inds = np.random.choice(all_inds, args.buffer_size)
    #     buffer = init_batch[init_inds]
    # elif args.buffer_init == "uniform":
    #     buffer = (torch.ones(args.buffer_size, *init_batch.size()[1:]) * .5).bernoulli()
    # else:
    #     raise ValueError("Invalid init")

    # if args.base_dist:
    #     model = EBM(net, init_mean)
    # else:
    #     model = EBM(net)

    dfs_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ebm_optimizer = torch.optim.Adam(dfs_model.parameters(), lr=args.ebm_lr, weight_decay=args.weight_decay)

    ema_ebm_model = copy.deepcopy(dfs_model)
    ema_dfs_model = copy.deepcopy(dfs_model)


    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
        ema_model.load_state_dict(d['ema_model'])
        #optimizer.load_state_dict(d['optimizer'])
        buffer = d['buffer']



    # move to cuda
    ebm_model.to(args.device)
    ema_ebm_model.to(args.device)
    dfs_model.to(args.device)
    ema_dfs_model.to(args.device)

    # get sampler
    # sampler = get_sampler(args)

    my_print(args.device)
    my_print(ebm_model)
    my_print(dfs_model)
    # my_print(buffer.size())
    # my_print(sampler)

    itr = 0
    best_val_ll = -np.inf
    hop_dists = []
    # all_inds = list(range(args.buffer_size))
    dfs_lr = args.lr
    ebm_lr = args.ebm_lr
    # init_dist = torch.distributions.Bernoulli(probs=init_mean.to(device))
    # reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(args.reinit_freq))
    test_ll_list = []
    while itr < args.n_iters:

        for x in train_loader:
            if itr < args.warmup_iters:
                ebm_lr = args.ebm_lr * float(itr) / args.warmup_iters
                for param_group in ebm_optimizer.param_groups:
                    param_group['lr'] = ebm_lr

            temp = args.start_temp
            temp_decay = args.end_temp/args.start_temp ** (1/args.dfs_per_ebm)

            ebm_model.eval()
            dfs_model.train()

            for i in args.dfs_per_ebm:
                dfs_samples = gen_samples(dfs_model, args)
                x1 = q_dist.sample(dfs_samples).to(args.device).long()

                x0 = torch.ones((B,D)).to(args.device).long() * M

                t = torch.rand((B,)).to(args.device)
                xt = x1.clone()
                mask = torch.rand((B,D)).to(args.device) < (1 - t[:, None])
                xt[mask] = x0[mask]

                dfs_loss, acc, weights = compute_loss(ebm_model, temp, dfs_model, q_dist, xt, x1, t, args)
                dfs_optimizer.zero_grad()
                dfs_loss.backward()
                dfs_optimizer.step()

                # update ema_model
                for p, ema_p in zip(dfs_model.parameters(), ema_dfs_model.parameters()):
                    ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

                temp = temp * temp_decay

            # if verbose:
            #     pbar.set_description(f'Epoch {itr} Iter {it} Loss {loss.item()}, Acc {acc}')

            # if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            #     eval_start_time = time.time()

            #     torch.save(dfs_model.state_dict(), f'{args.ckpt_path}/dfs_model_{epoch}.pt')

            #     log_entry = {'epoch':None,'timestamp':None}
                
            #     log_entry['loss'] = loss.item()
            #     log_entry['acc'] = acc
            #     torch.save(ema_dfs_model.state_dict(), f'{args.ckpt_path}/ema_dfs_model_{epoch}.pt')

            #     plot(f'{args.sample_path}/q_samples_{epoch}_first_ten_types_{q_dist.get_last_is_empirical()[0:10].cpu().numpy()}.png', torch.tensor(x1).float())
            #     samples = gen_samples(dfs_model, args, 100)
            #     plot(f'{args.sample_path}/samples_{epoch}.png', torch.tensor(samples).float())
            #     ema_samples = gen_samples(ema_dfs_model, args, 100)
            #     plot(f'{args.sample_path}/ema_samples_{epoch}.png', torch.tensor(ema_samples).float())
            #     eval_end_time = time.time()
            #     output_dir = f'{args.plot_path}/weights_histogram_{epoch}.png'
            #     if not os.path.exists(output_dir):
            #         plot_weight_histogram(weights, output_dir=output_dir)
            #     eval_time = eval_end_time - eval_start_time
            #     cum_eval_time += eval_time
            #     timestamp = time.time() - cum_eval_time - start_time

            #     log(args, log_entry, epoch, timestamp)

            ebm_model.train()
            dfs_model.eval()

            x = preprocess(x[0].to(device).requires_grad_())

            # choose random inds from buffer
            # buffer_inds = sorted(np.random.choice(all_inds, args.batch_size, replace=False))
            # x_buffer = buffer[buffer_inds].to(device)
            # reinit = reinit_dist.sample((args.batch_size,)).to(device)
            # x_reinit = init_dist.sample((args.batch_size,)).to(device)
            # x_fake = x_reinit * reinit[:, None] + x_buffer * (1. - reinit[:, None])

            x_fake = gen_samples(dfs_model, args)

            # hops = []  # keep track of how much the sampelr moves particles around
            # st = time.time()
            # for k in range(args.sampling_steps):
            #     x_fake_new = sampler.step(x_fake.detach(), model).detach()
            #     h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
            #     hops.append(h)
            #     x_fake = x_fake_new
            # st = time.time() - st
            # hop_dists.append(np.mean(hops))

            # update buffer
            # buffer[buffer_inds] = x_fake.detach().cpu()

            logp_real = ebm_model(x).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg = 0.0

            logp_fake = ebm_model(x_fake).squeeze()

            obj = logp_real.mean() - logp_fake.mean()
            ebm_loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

            optimizer.zero_grad()
            ebm_loss.backward()
            optimizer.step()
            

            # update ema_model
            for p, ema_p in zip(ebm_model.parameters(), ema_ebm_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if itr % args.print_every == 0:
                my_print("({}) | ({}/iter) cur ebm_lr = {:.4f} |log p(real) = {:.4f}, "
                         "log p(fake) = {:.4f}, obj = {:.4f}, dfs_loss = {:.4f}".format(itr, st, ebm_lr, logp_real.mean().item(),
                                                                                     logp_fake.mean().item(), obj.item(),
                                                                                     dfs_loss.item()))
            if itr % args.viz_every == 0:
                plot("{}/data_{}.png".format(args.save_dir, itr), x.detach().cpu())
                plot("{}/dfs_samples_{}.png".format(args.save_dir, itr), x_fake)


            if (itr + 1) % args.eval_every == 0:
                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_ebm_model, init_dist, sampler,
                                                                            train_loader, val_loader, test_loader,
                                                                            preprocess, device,
                                                                            args.eval_sampling_steps,
                                                                            args.test_batch_size)
                my_print("EMA Train log-likelihood ({}): {}".format(itr, train_ll.item()))
                my_print("EMA Valid log-likelihood ({}): {}".format(itr, val_ll.item()))
                my_print("EMA Test log-likelihood ({}): {}".format(itr, test_ll.item()))
                test_ll_list.append(test_ll.item())
                for _i, _x in enumerate(ais_samples):
                    plot("{}/EMA_sample_{}_{}_{}_{}_{}.png".format(args.save_dir, args.dataset_name,args.sampler,args.step_size,itr, _i), _x)

                ebm_model.cpu()
                d = {}
                d['ebm_model'] = ebm_model.state_dict()
                d['ema_ebm_model'] = ema_ebm_model.state_dict()
                d['buffer'] = buffer
                d['optimizer'] = optimizer.state_dict()
                if val_ll.item() > 0:
                    exit()
                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    my_print("Best valid likelihood")
                    torch.save(d, "{}/best_ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.sampler,args.step_size))
                else:
                    torch.save(d, "{}/ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.sampler,args.step_size))

                ebm_model.to(device)

            itr += 1
    np.save("{}/test_ll_{}_{}_{}.npy".format(args.save_dir,args.dataset_name,args.sampler,args.step_size),test_ll_list)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="./figs/ebm")
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--ckpt_path', type=str, default=None)
    # data generation
    parser.add_argument('--n_out', type=int, default=3)     # potts
    # models
    parser.add_argument('--ebm_model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', action='store_true')
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--ema', type=float, default=0.999)
    # mcmc
    parser.add_argument('--sampler', type=str, default='gibbs')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--reinit_freq', type=float, default=0.0)
    parser.add_argument('--eval_sampling_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    parser.add_argument('--step_size', type=float, default=0.08)
    # training
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--warmup_iters', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--ebm_lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=.0)

    args = parser.parse_args()
    args.device = device
    main(args)
