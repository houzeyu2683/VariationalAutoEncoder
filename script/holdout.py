
import data
import network

'''
Example
    `tabulation.data`
    `tabulation.train`
    `tabulation.test`
'''
tabulation = data.tabulation(path="../##Data##/celebfacesattribute/csv/index.csv")
tabulation.read()

'''
Example
    `tabulation.grid.train`
    `tabulation.grid.exam`
'''
tabulation.hold(ratio=0.2)

'''
Example
    `dataset['train'].get(0)`
    `dataset['exam'].get(0)`
    `dataset['test'].get(0)`
'''
dataset = {
    'train' : data.dataset(tabulation.grid.train),
    'exam'  : data.dataset(tabulation.grid.exam) ,
    'test'  : data.dataset(tabulation.test)      ,
}

'''
Example
    `next(iter(loader.train))`
    `next(iter(loader.exam))`
    `next(iter(loader.test))`
    `loader.save(what='batch', folder='./')`
'''
loader = data.loader(train=dataset['train'], exam=dataset['exam'], test=dataset['test'], batch=64)

'''
Example
    `machine.learn(loader=loader.train)`
'''
model     = network.model()
optimizer = network.optimizer.adam(model)
machine   = network.machine(model, optimizer=optimizer, folder="LOG")

for i in range(5):

    machine.learn(loader=loader.train)
    machine.save(what='checkpoint')
    machine.evaluate(loader=loader.train, title='train', number=1)
    machine.evaluate(loader=loader.exam , title='test', number=1)
    machine.update(what='checkpoint')
    machine.update(what='schedule')
    pass
