import sys

img_path = "images"
old_files = ['database', 'test', 'train']
for file in old_files:
    with open('{}.txt'.format(file), 'r') as f:
        labels = f.readlines()

    with open('{}_new.txt'.format(file), 'w') as out:
        total = len(labels)
        for c, label in enumerate(labels):
            base_label = label.strip().split('/')[-1]

            out.write("images/{}\n".format(base_label))
            sys.stdout.write("processed: %d/%d\r" % (c, total))
            sys.stdout.flush()