
##
##
import data
import network

tabulation = data.tabulation(path="./celebfaces-attribute-sample/csv/index.csv")
tabulation.read()
tabulation.data['link'] = ["./celebfaces-attribute-sample/jpg/" + i for i in tabulation.data['image']]
tabulation.distinct(what='data')

'''
Example
    `tabulation.data`
    `tabulation.train`
    `tabulation.exam`
    `tabulation.test`
'''

# tabulation.hold(ratio=0.2)
# '''
# Example
#     `tabulation.grid.train`
#     `tabulation.grid.exam`
# '''

dataset = {
    'train' : data.dataset(tabulation.train),
    'exam'  : data.dataset(tabulation.exam) ,
    'test'  : data.dataset(tabulation.test) ,
}
'''
Example
    `dataset['train'].get(0)`
    `dataset['exam'].get(0)`
    `dataset['test'].get(0)`
'''

loader = data.loader(train=dataset['train'], exam=dataset['exam'], test=dataset['test'], batch=144)
'''
Example
    `next(iter(loader.train))`
    `next(iter(loader.exam))`
    `next(iter(loader.test))`
    `loader.save(what='batch', folder='./')`
'''

model     = network.model()
optimizer = network.optimizer.adam(model)
machine   = network.machine(model, optimizer=optimizer, folder="LOG")
'''
Example
    `machine.learn(loader=loader.train)`
'''

for i in range(25):

    machine.learn(loader=loader.train)
    machine.save(what='checkpoint')
    machine.evaluate(loader=loader.train, title='train', number=1)
    machine.evaluate(loader=loader.exam , title='test', number=1)
    machine.update(what='checkpoint')
    machine.update(what='schedule')
    pass
