import os
import torch
import random
import numpy as np
from tqdm import tqdm
from model import RobertaMultiTaskClassifier
from transformers import RobertaTokenizerFast, AdamW
from config import get_config
from data_process import get_main_task_train_dataloader, \
    get_main_task_eval_dataloader, \
    get_main_task_test_dataloader, \
    get_category_task_train_dataloader, \
    get_category_task_eval_dataloader, \
    get_category_task_test_dataloader, \
    get_attach_task_train_dataloader, \
    get_attach_task_eval_dataloader, \
    get_attach_task_test_dataloader, \
	get_all_category

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')

def seed_everything(seed):
    print('Setting global seed to {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args):
    seed_everything(args['seed'])
    category = get_all_category(args)
    tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])

    model = RobertaMultiTaskClassifier.from_pretrained(args['pretrained_model'])
    model.to(torch.device("cuda"))

    main_task_train_dataloader = get_main_task_train_dataloader(args, tokenizer)
    category_task_train_dataloader = get_category_task_train_dataloader(args, tokenizer)
    attach_task_train_dataloader = get_attach_task_train_dataloader(args, tokenizer)

    save_dir = os.path.join(args['model_dir'], 'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    step_cnt = 0

    optimizer = AdamW(model.parameters(), lr=args['lr'])

    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args['n_epoch']):
        all_batch = []
        for batch in main_task_train_dataloader:
            all_batch.append(('main', batch))
        for batch in category_task_train_dataloader:
            all_batch.append(('category', batch))
        for batch in attach_task_train_dataloader:
            all_batch.append(('attach', batch))
        random.shuffle(all_batch)
        m_acc, m_loss, c_acc, c_loss, a_acc, a_loss = 0, 0, 0, 0, 0, 0

        pbar = tqdm(all_batch, position=0, desc='Epoch {}'.format(epoch))
        n_main_task_correct, main_task_total = 0, 0
        n_category_task_correct, category_task_total = 0, 0
        n_attach_task_correct, attach_task_total = 0, 0
        for task, batch in pbar:
            if task == 'main':
                optimizer.zero_grad()
                inputs, numeral_position, start_position, end_position = batch
                start_logits, end_logits = model(inputs.cuda(), numeral_position, task)
                start_logits = start_logits.cpu()
                end_logits = end_logits.cpu()
                start_loss = loss_fn(start_logits, start_position)
                end_loss = loss_fn(end_logits, end_position)
                loss = (start_loss + end_loss) / 2
                start_pred = torch.argmax(start_logits, dim=-1)
                end_pred = torch.argmax(end_logits, dim=-1)
                for start_p, end_p, start_r, end_r, input_ids in zip(start_pred, end_pred, start_position, end_position, inputs):
                    if start_p >= end_p:
                        p = 'None'
                    else:
                        p = tokenizer.decode(input_ids[start_p: end_p])
                    
                    if start_r >= end_r:
                        r = 'None'
                    else:
                        r = tokenizer.decode(input_ids[start_r: end_r])
                    if p == r:
                        n_main_task_correct += 1
                    main_task_total += 1

                m_acc = n_main_task_correct / main_task_total
                m_loss = loss.item()
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
                step_cnt += 1

                loss.backward()
                optimizer.step()

            if task == 'category':
                optimizer.zero_grad()

                inputs, numeral_position, category_id = batch
                logits = model(inputs.cuda(), numeral_position, task).cpu()
                loss = loss_fn(logits, category_id)
                for logit, label in zip(logits, category_id):
                    max_arg = 0
                    for idx in range(len(logit)):
                        if logit[idx].item() > logit[max_arg].item():
                            max_arg = idx
                    if max_arg == label.item():
                        n_category_task_correct += 1
                    category_task_total += 1
                c_acc = n_category_task_correct / category_task_total
                c_loss = loss.item()
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
                step_cnt += 1

                loss.backward()
                optimizer.step()

            if task == 'attach':
                optimizer.zero_grad()

                inputs, numeral_position, sign_id = batch
                logits = model(inputs.cuda(), numeral_position, task).cpu()
                loss = loss_fn(logits, sign_id)
                for logit, label in zip(logits, sign_id):
                    max_arg = 0
                    for idx in range(len(logit)):
                        if logit[idx].item() > logit[max_arg].item():
                            max_arg = idx
                    if max_arg == label.item():
                        n_attach_task_correct += 1
                    attach_task_total += 1
                a_acc = n_attach_task_correct / attach_task_total
                a_loss = loss.item()
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
                step_cnt += 1

                loss.backward()
                optimizer.step() 

            if type(args['saving_steps']) != type(None) and step_cnt % args['saving_steps'] == 0:
                ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_name = 'step_ckpt_{}'.format(step_cnt // args['saving_steps'])
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                model.save_pretrained(ckpt_path)

        if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
            ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_name = 'epoch_ckpt_{}'.format(epoch // args['saving_epochs'])
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            model.save_pretrained(ckpt_path)
    
    model.save_pretrained(save_dir)

def test(args):
    seed_everything(args['seed'])

    tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
    model = RobertaMultiTaskClassifier.from_pretrained(args['testing_model'])
    model.to('cuda')
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    main_task_test_dataloader = get_main_task_test_dataloader(args, tokenizer)
    category_task_test_dataloader = get_category_task_test_dataloader(args, tokenizer)
    attach_task_test_dataloader = get_attach_task_test_dataloader(args, tokenizer)
    
    n_main_task_correct, main_task_total = 0, 0
    n_category_task_correct, category_task_total = 0, 0
    n_attach_task_correct, attach_task_total = 0, 0
    with torch.no_grad(): 
        all_batch = []
        for batch in main_task_test_dataloader:
            all_batch.append(('main', batch))
        for batch in category_task_test_dataloader:
            all_batch.append(('category', batch))
        for batch in attach_task_test_dataloader:
            all_batch.append(('attach', batch))
        pbar = tqdm(all_batch, position=0, desc='inference')
        m_acc, c_acc, a_acc = 0, 0, 0

        for task, batch in pbar:
            if task == 'main':
                inputs, numeral_position, start_position, end_position = batch
                start_logits, end_logits = model(inputs.cuda(), numeral_position, task)
                start_logits = start_logits.cpu()
                end_logits = end_logits.cpu()
                start_loss = loss_fn(start_logits, start_position)
                end_loss = loss_fn(end_logits, end_position)
                loss = (start_loss + end_loss) / 2
                start_pred = torch.argmax(start_logits, dim=-1)
                end_pred = torch.argmax(end_logits, dim=-1)
                for start_p, end_p, start_r, end_r, input_ids in zip(start_pred, end_pred, start_position, end_position, inputs):
                    if start_p >= end_p:
                        p = 'None'
                    else:
                        p = tokenizer.decode(input_ids[start_p: end_p])
                    
                    if start_r >= end_r:
                        r = 'None'
                    else:
                        r = tokenizer.decode(input_ids[start_r: end_r])
                    if p == r:
                        n_main_task_correct += 1

                    main_task_total += 1
                m_acc = n_main_task_correct / main_task_total
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
            if task == 'category':
                inputs, numeral_position, category_id = batch
                logits = model(inputs.cuda(), numeral_position, task).cpu()
                loss = loss_fn(logits, category_id)
                for logit, label in zip(logits, category_id):
                    max_arg = 0
                    for idx in range(len(logit)):
                        if logit[idx].item() > logit[max_arg].item():
                            max_arg = idx
                    if max_arg == label.item():
                        n_category_task_correct += 1
                    category_task_total += 1
                c_acc = n_category_task_correct / category_task_total
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
            if task == 'attach':
                inputs, numeral_position, sign_id = batch
                logits = model(inputs.cuda(), numeral_position, task).cpu()
                loss = loss_fn(logits, sign_id)
                for logit, label in zip(logits, sign_id):
                    max_arg = 0
                    for idx in range(len(logit)):
                        if logit[idx].item() > logit[max_arg].item():
                            max_arg = idx
                    if max_arg == label.item():
                        n_attach_task_correct += 1
                    attach_task_total += 1
                a_acc = n_attach_task_correct / attach_task_total
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})

    main_task_eval_dataloader = get_main_task_eval_dataloader(args, tokenizer)
    category_task_eval_dataloader = get_category_task_eval_dataloader(args, tokenizer)
    attach_task_eval_dataloader = get_attach_task_eval_dataloader(args, tokenizer)
    
    n_main_task_correct, main_task_total = 0, 0
    n_category_task_correct, category_task_total = 0, 0
    n_attach_task_correct, attach_task_total = 0, 0
    with torch.no_grad(): 
        all_batch = []
        for batch in main_task_eval_dataloader:
            all_batch.append(('main', batch))
        for batch in category_task_eval_dataloader:
            all_batch.append(('category', batch))
        for batch in attach_task_eval_dataloader:
            all_batch.append(('attach', batch))
        pbar = tqdm(all_batch, position=0, desc='evaluate')
        m_acc, c_acc, a_acc = 0, 0, 0

        for task, batch in pbar:
            if task == 'main':
                inputs, numeral_position, start_position, end_position = batch
                start_logits, end_logits = model(inputs.cuda(), numeral_position, task)
                start_logits = start_logits.cpu()
                end_logits = end_logits.cpu()
                start_loss = loss_fn(start_logits, start_position)
                end_loss = loss_fn(end_logits, end_position)
                loss = (start_loss + end_loss) / 2
                start_pred = torch.argmax(start_logits, dim=-1)
                end_pred = torch.argmax(end_logits, dim=-1)
                for start_p, end_p, start_r, end_r, input_ids in zip(start_pred, end_pred, start_position, end_position, inputs):
                    if start_p >= end_p:
                        p = 'None'
                    else:
                        p = tokenizer.decode(input_ids[start_p: end_p])
                    
                    if start_r >= end_r:
                        r = 'None'
                    else:
                        r = tokenizer.decode(input_ids[start_r: end_r])
                    if p == r:
                        n_main_task_correct += 1

                    main_task_total += 1
                m_acc = n_main_task_correct / main_task_total
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
            if task == 'category':
                inputs, numeral_position, category_id = batch
                logits = model(inputs.cuda(), numeral_position, task).cpu()
                loss = loss_fn(logits, category_id)
                for logit, label in zip(logits, category_id):
                    max_arg = 0
                    for idx in range(len(logit)):
                        if logit[idx].item() > logit[max_arg].item():
                            max_arg = idx
                    if max_arg == label.item():
                        n_category_task_correct += 1
                    category_task_total += 1
                c_acc = n_category_task_correct / category_task_total
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})
            if task == 'attach':
                inputs, numeral_position, sign_id = batch
                logits = model(inputs.cuda(), numeral_position, task).cpu()
                loss = loss_fn(logits, sign_id)
                for logit, label in zip(logits, sign_id):
                    max_arg = 0
                    for idx in range(len(logit)):
                        if logit[idx].item() > logit[max_arg].item():
                            max_arg = idx
                    if max_arg == label.item():
                        n_attach_task_correct += 1
                    attach_task_total += 1
                a_acc = n_attach_task_correct / attach_task_total
                pbar.set_postfix({'m_acc': '{:.3f}%'.format(m_acc * 100), 'c_acc': '{:.3f}%'.format(c_acc * 100), 'a_acc': '{:.3f}%'.format(a_acc * 100)})

if __name__ == '__main__':
    args = get_config()
    if args['mode'] == 'train':
        train(args)
    if args['mode'] == 'test':
        test(args)
