
import pandas, os

'''
All of data define the train mode.
'''
folder = os.path.join(os.environ['HOME'], "Public/##Data##/celebfacesattribute/jpg/")
group = os.listdir(folder)
link = [os.path.join(folder, i) for i in group]
table = pandas.DataFrame({'image':group, "link":link, 'mode':"train"})
pass

'''
Sample same data define test mode.
'''
table = table.sample(len(table)).reset_index(drop=True)
size = int(len(table) * 0.1)
table.loc[range(size), 'mode'] = 'test'
pass

'''
Save the table.
'''
record = os.path.join(os.environ['HOME'], "Public/##Data##/celebfacesattribute/csv/")
os.makedirs(record, exist_ok=True)
table.to_csv(os.path.join(record, 'index.csv'), index=False)
pass

