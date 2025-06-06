import warnings

warnings.filterwarnings("ignore")

from utils.MvDataloaders_temp import Get_dataloaders
from utils.MvLoad_models import load
from sklearn.cluster import KMeans
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import numpy as np
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(15)

NMI_c = []
NMI_cz = []
ACC_c = []
ACC_cz = []

datasets = ['Multi-COIL-10',        #0
            'Multi-COIL-20',        #1
            'Multi-MNIST',          #2
            'Multi-FMNIST',         #3
            'MNIST-USPS',           #4
            'DHA',                  #5
            ]

settings = [[1, 32], [1, 32], [1, 32], [1, 64], [1, 64], [1, 64], [1,64],[1,64],[1,64],[1,64],[1,64],[1,64],[1,64],[1,64],[1,64]]  # share autoencoder, Batch_size
iters_to_add_capacity = [25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000]


for d in [1]:  # datasets index
    DATA = datasets[d]
    share = settings[d][0]
    Batch_size = settings[d][1]
    iters_add_capacity = iters_to_add_capacity[d]
    Epochs = 500
    lr = 5e-4
    Net = 'C'  # CNN
    hidden_dim = 256
    z_variables = 10    # view-peculiar variable
    krpa = 1
    runs = 1
    TEST = False

    for beta in [30]:
        for capacity in [5]:
            ACCc = 0
            NMIc = 0
            ARIc = 0
            PURc = 0
            ACCcz = 0
            NMIcz = 0
            ARIcz = 0
            PURcz = 0
            for i in range(runs):
                model_name = DATA + '.pt'
                print(model_name)
                print('Run:' + str(i))

                if Net == 'C':
                    train_loader, test_loader, view_num, n_clusters, size = Get_dataloaders(batch_size=Batch_size,
                                                                               DATANAME=DATA + '.mat')  # 64
                    print('Iters:' + str(size / Batch_size * Epochs))
                    print('size', size)
                    print('Batch_size', Batch_size)
                    print('epoch', Epochs)
                    # Define the capacities
                    # Continuous channels
                    # Discrete channels
                    # iters_add_capacity = size/Batch_size*Epochs
                    cont_capacity = [capacity, beta, iters_add_capacity]
                    disc_capacity = [np.log(n_clusters), beta, iters_add_capacity]

                latent_spec = {'cont': z_variables,
                               'disc': [n_clusters]}
                use_cuda = torch.cuda.is_available()
                # use_cuda = False
                print("cuda is available?")
                print(use_cuda)

                # img_size=(3, 64, 64)
                # img_size=(3, 32, 32)
                img_size = (1, 32, 32)

                # training mode
                if TEST == False:
                    # Build a model
                    from multi_vae.MvModels import VAE
                    if Net=='C':
                        model = VAE(latent_spec=latent_spec, img_size=img_size,
                                view_num=view_num, use_cuda=use_cuda,
                                Network=Net, hidden_dim=hidden_dim, shareAE=share, min_k=krpa)
                    if use_cuda:
                        model.cuda()
                    print(use_cuda)

                    print(model)

                    # Train the model
                    from torch import optim

                    # Build optimizer
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    from multi_vae.MvTraining import Trainer

                    # Build a trainer
                    trainer = Trainer(model, optimizer, model_name,
                                      cont_capacity=cont_capacity,
                                      disc_capacity=disc_capacity, view_num=view_num, use_cuda=use_cuda, DATA=DATA)

                    # Train model for Epochs
                    trainer.train(train_loader, epochs=Epochs, net=Net)
                    torch.save(trainer.model.state_dict(), './models/' + model_name)
                    # print(trainer.model.state_dict())
                    print('save model?')
                    TEST = True

                # testing mode
                if TEST == True:
                    path_to_model_folder = './models/' + model_name
                    batch_size_test = 140000  # max number to cover all dataset.
                    if Net == 'C':
                        train_loader, test_loader, view_num, n_clusters, _ = Get_dataloaders(batch_size=batch_size_test,
                                                                                DATANAME=DATA + '.mat')
                        model = load(latent_spec=latent_spec,
                                     path=path_to_model_folder,
                                     view_num=view_num,
                                     img_size=img_size,
                                     Network=Net,
                                     hid=hidden_dim, shareAE=share)

                    # Print the latent distribution info
                    print("latent distribution info", model.MvLatent_spec)

                    # Print model architecture
                    print(model)
                    from torch.autograd import Variable

                    for batch_idx, Data in enumerate(test_loader):
                        break

                    data = Data[0:-1]
                    labels = Data[-1]
                    inputs = []
                    for i in range(view_num):
                        inputs.append(Variable(data[i]))
                    # inputs.to('cuda')
                    encodings = model.encode(inputs)

                    import Nmetrics

                    kmeans = KMeans(n_clusters=n_clusters, n_init=100)

                    # Discrete encodings, view-common variable
                    x = encodings['disc'][0].cpu().detach().data.numpy()
                    multiview_z = []
                    multiview_cz = []
                    for i in range(view_num):
                        name = 'cont' + str(i + 1)

                        x_c = encodings[name][0].cpu().detach().data.numpy()  # z
                        xi = min_max_scaler.fit_transform(x_c)  # scale to [0,1]
                        multiview_z.append(np.concatenate([xi, x], axis=1))  # z + c
                        multiview_cz.append(xi)
                        print(multiview_z[-1].shape)
                        print(multiview_z[-1][0])
                    y = labels.cpu().detach().data.numpy()

                    p = kmeans.fit_predict(x)
                    print('k-means on C')
                    print(x.shape)
                    from Nmetrics import test

                    test(y, p)

                    p = x.argmax(1)
                    print('Multi-VAE-C: y = C.argmax(1)')
                    test(y, p)
                    ACCc += Nmetrics.acc(y, p)
                    NMIc += Nmetrics.nmi(y, p)
                    ARIc += Nmetrics.ari(y, p)
                    PURc += Nmetrics.purity(y, p)

                    X_all = np.concatenate(multiview_cz, axis=1)
                    p = kmeans.fit_predict(X_all)
                    print('k-means on [z1, z2, ..., zV]')
                    print(X_all.shape)
                    test(y, p)
                    print('k-means on [zv]\nk-means on [C, zv]')
                    print(multiview_cz[0].shape, multiview_z[0].shape)
                    for i in range(view_num):
                        name = 'cont' + str(i + 1)
                        x_cz = encodings[name][0].cpu().detach().data.numpy()
                        x_Conz = multiview_z[i]
                        p = kmeans.fit_predict(x_cz)
                        test(y, p)
                        p = kmeans.fit_predict(x_Conz)
                        test(y, p)
                        print('\n')

                    multiview_cz.append(x)
                    X_all = np.concatenate(multiview_cz, axis=1)
                    p = kmeans.fit_predict(X_all)
                    # scio.savemat('./viz/' + str(Epochs) + '.mat', {'Z': X_all, 'Y': y, 'P': p})
                    print('Multi-VAE-CZ: k-means on [C, z1, z2, ..., zV]')
                    print(X_all.shape)
                    test(y, p)
                    ACCcz += Nmetrics.acc(y, p)
                    NMIcz += Nmetrics.nmi(y, p)
                    ARIcz += Nmetrics.ari(y, p)
                    PURcz += Nmetrics.purity(y, p)

                    # for beta
                    TEST = False

            print('Multi-VAE-C:', ACCc / runs, NMIc / runs, ARIc / runs, PURc / runs)
            print('Multi-VAE-CZ:', ACCcz / runs, NMIcz / runs, ARIcz / runs, PURcz / runs)
            # np.save('Cmetics.npy', [ACCc/runs, NMIc/runs, ARIc/runs, PURc/runs])
            # np.save('CZmetics.npy', [ACCcz/runs, NMIcz/runs, ARIcz/runs, PURcz/runs])
            NMI_c.append(NMIc / runs)
            NMI_cz.append(NMIcz / runs)
            ACC_c.append(ACCc / runs)
            ACC_cz.append(ACCcz / runs)
    # print(NMI_c)
    # np.save('result/NMI_c.npy', NMI_c)
    print(NMI_cz)
    # np.save('result/NMI_cz.npy', NMI_cz)
    # print(ACC_c)
    # np.save('result/ACC_c.npy', ACC_c)
    print(ACC_cz)
    # np.save('result/ACC_cz.npy', ACC_cz)
