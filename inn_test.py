import visdom
import torch as t
import math
import GPF


def s(x): return x.squeeze().cpu()


def std(x): return 1/math.sqrt(2*math.pi) * t.exp(-x**2/2)


def subnet_fc(c_in, c_out):
    return t.nn.Sequential(t.nn.Linear(c_in, 100), t.nn.ReLU(), t.nn.Linear(100,  c_out))


viz = visdom.Visdom()
size = 10**4
n_conDim = 10
data = t.randn((size, 1))/4
con = t.zeros((size, n_conDim))
for i in range(1, 6):
    data = t.cat((data,  t.randn((size, 1))/4+i), dim=0)
    con = t.cat((con, t.zeros(size, n_conDim)), dim=0)
data = (data-t.mean(data)).cuda()
con = con.cuda()

xV = t.arange(-3, 3, 0.1).unsqueeze(1).cuda()
net = GPF.Net(n_blocks=4, n_dim=1, ls=30, n_condim=n_conDim, subnet_constructor=subnet_fc).cuda()

win1 = viz.histogram(X=data.squeeze(), opts=dict(numbins=50, title='ground truth'))
z = net(x=xV, y=t.zeros(len(xV), n_conDim).cuda(), rev=False)
win2 = viz.line(X=s(xV), Y=s(z), opts=dict(title='flow'))
win3 = viz.line(X=s(xV), Y=s(std(z) * t.exp(net.log_jacobian)), opts=dict(title='estimated density'))
optimizer = t.optim.Adam(net.parameters(), lr=0.01, weight_decay=10 ** -6)

for i in range(1000):
    z = net.forward(x=data, y=con, rev=False)
    grad = net.log_jacobian
    loss = t.mean(z ** 2 / 2 - grad)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(loss)
        z = net(x=xV, y=t.zeros(len(xV), n_conDim).cuda(), rev=False)

        viz.line(X=s(xV), Y=s(z), win=win2, opts=dict(title='flow'))
        viz.line(X=s(xV), Y=s(std(z) * t.exp(net.log_jacobian)), win=win3, opts=dict(title='estimated density'))
