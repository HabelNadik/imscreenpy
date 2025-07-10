from os.path import join, isfile, isdir
from os import listdir

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class Visualizer:
    
    def __init__(self, input_table, vis_folder, row_id='Metadata_row', col_id='Metadata_column', field_id='Metadata_field', x_id='AreaShape_Center_X', y_id='AreaShape_Center_Y'):
        self.table = input_table
        
        self.field_id = field_id
        
        self.row_id = row_id
        self.col_id = col_id
        
        self.x_id = x_id
        self.y_id = y_id
        
        self.nametemplate = 'r{}c{}f{}p01-ch4sk1fk1fl1_color.tiff'
        self.nametemplate2 = 'r{}c{}f{}p01-ch4sk1fk1fl1_gamma.tiff'
        self.nametemplate3 = 'r{}c{}f{}p01-ch3sk1fk1fl1_color.tiff'
        self.nametemplate4 = 'r{}c{}f{}p01-ch1sk1fk1fl1_gamma.tiff'
        self.nametemplate5 = 'r{}c{}f{}p01-ch1sk1fk1fl1_color.tiff'

        self.all_nametemplates = [self.nametemplate, self.nametemplate2, self.nametemplate3, self.nametemplate4, self.nametemplate5]

        self.vis_folder = vis_folder
        
    def create_rectangle_borders(self, y, x, def_size=70, y_shape=1080, x_shape=1080, return_relative_yx=False):
        y = round(y)
        x = round(x)
        y_min = y - round(def_size/2)
        rel_y = round(def_size/2)
        rel_x = round(def_size/2)
        if y_min < 0:
            rel_y += (y_min)
            y_min = 0
        y_max = y + round(def_size/2)
        if y_max > y_shape:
            #rel_y += (y_shape - y_max)
            y_max = y_shape
        x_min = x - round(def_size/2)
        if x_min < 0:
            rel_x += x_min
            x_min = 0
        x_max = x + round(def_size/2)
        if x_max > x_shape:
            #rel_x += (x_shape + x_max)
            x_max = x_shape
        if return_relative_yx:
            return y_min, y_max, x_min, x_max, rel_y, rel_x
        return y_min, y_max, x_min, x_max
        
    def make_path(self, row, col, field):
        all_names = [self.make_name(f, row, col, field) for f in self.all_nametemplates]
        all_paths = [join(self.vis_folder, f) for f in all_names]
        for path in all_paths:
            if isfile(path):
                return path
        subdirectories = [join(self.vis_folder, f) for f in listdir(self.vis_folder) if isdir(join(self.vis_folder, f))]
        for subdir in subdirectories:
            for name in all_names:
                if isfile(join(subdir, name)):
                    im_path = join(subdir, name)
                    return im_path
        else:
            print('image with pattern {} not found'.format(all_names[0][:12]))
            return None
        
    def make_name(self, template, row, col, field):
        if len(str(row)) > 1:
            row = str(row)
        else:
            row = '0' + str(row)
        if len(str(col)) > 1:
            col = str(col)
        else:
            col = '0' + str(col)
        if len(str(field)) > 1:
            field = str(field)
        else:
            field = '0' + str(field)
        return template.format(row, col, field)

    def visualize_single_cell(self, table_index, def_size=70, return_xy=False):
        row_oi = self.table.iloc[table_index,:]
        x_pos = row_oi[self.x_id]
        y_pos = row_oi[self.y_id]
        if (np.isnan(y_pos)) or np.isnan(x_pos):
            out_im = (np.ones((def_size,def_size,3)) * 255).astype(np.uint8)
            if return_xy:
                return out_im, 0, 0
            else:
                return out_im
        row = str(row_oi[self.row_id])
        col = str(row_oi[self.col_id])
        field = str(row_oi[self.field_id])
        
        if len(row) == 1:
            row = '0' + str(row)
        if len(col) == 1:
            col = '0' + str(col)
        if len(field) == 1:
            field = '0' + str(field)
        im_path = self.make_path(row, col, field)
        #print(isfile(im_path))
        im = cv2.imread(im_path)
        if (im is None):
            print('Image at table index {} and row: {} col: {} field: {} not found'.format(table_index, row, col, field))
            out_im = (np.ones((def_size,def_size,3)) * 255).astype(np.uint8)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #y_min, y_max, x_min, x_max = self.create_rectangle_borders(y_pos, x_pos)
            y_min, y_max, x_min, x_max, y_rel, x_rel = self.create_rectangle_borders(y_pos, x_pos, return_relative_yx=True, def_size=def_size)
            #print('Working with coords x={}, y={}'.format(x_pos, y_pos))
            out_im = im[y_min:y_max,x_min:x_max]
            out_im = self.draw_cross(out_im, y_rel, x_rel, def_size=def_size)
        if return_xy:
            return out_im, y_pos, x_pos
        return out_im
    
    def draw_cross(self, mini_im, y, x, half_width=1, half_length=4, def_size=70):
        ## vertical part
        width = 2 * half_width
        vert_width = width
        hor_length = width
        length = 2 * half_length
        cross_top = y - half_length
        cross_bot = y + half_length
        cross_vert_left = x-half_width
        cross_vert_right = x+half_width
        cross_left = x - half_length
        cross_right = x + half_length
        cross_hor_top = y - half_width
        cross_hor_bot = y + half_width
        if cross_top < 0:
            vert_length = length + cross_top
            cross_top = 0
            if cross_hor_top < 0:
                hor_length = hor_length + cross_hor_top
                cross_hor_top = 0
        elif cross_bot > mini_im.shape[0]:
            vert_length = length - (cross_bot - mini_im.shape[0])
            cross_bot = mini_im.shape[0]
            if cross_hor_bot> mini_im.shape[0]:
                hor_length = hor_length - (cross_hor_bot - mini_im.shape[0])
                cross_hor_bot = mini_im.shape[0]
        else:
            vert_length = length
        if cross_left < 0:
            hor_width = length + cross_left
            cross_left = 0
            if cross_vert_left < 0:
                cross_vert_left = 0
                vert_width = vert_width - cross_left - 1
        elif cross_right > mini_im.shape[1]:
            hor_width = length - (cross_right - mini_im.shape[1])
            cross_right = mini_im.shape[1]
            if cross_vert_right > mini_im.shape[1]:
                cross_vert_right =  mini_im.shape[1]
                vert_width = vert_width - (cross_right - mini_im.shape[1]) - 1
        else:
            hor_width = length
        if cross_left < 0:
            cross_left = 0
        if np.amax(mini_im) > 1.1:
            use_num = 255
        else:
            use_num = 1.
        mini_im[cross_top:cross_bot,cross_vert_left:cross_vert_right] = np.ones((vert_length,vert_width,3)).astype(mini_im.dtype) * use_num
        mini_im[cross_hor_top:cross_hor_bot,cross_left:cross_right] = np.ones((hor_length,hor_width,3)).astype(mini_im.dtype) * use_num
        return mini_im



    def show_single_cell(self, table_index):
        im = self.visualize_single_cell(table_index)
        plt.imshow(im)
        plt.show()
        
    def get_field_image(self, row, col, field):
        field_im = cv2.imread(self.make_path(row, col, field), -1)
        field_im = cv2.cvtColor(field_im, cv2.COLOR_BGR2RGB)
        return field_im

    def make_well_image(self, row, col, out_res=4080, layout_path='/mnt/d/develop/PreciseOncology/extract_aml/sample_annotations/imaging_layout_std.txt'):
        layout = np.loadtxt(layout_path)
        num_fields = int(np.amax(layout))
        all_images = [self.get_field_image(row, col, f) for f in range(1, num_fields+1)]
        out_im = np.zeros((out_res, out_res, 3), dtype=np.uint8)
        target_field_resolution = (int(float(out_res) / float(layout.shape[0])),  int(float(out_res) / float(layout.shape[1])), 3)
        for i in range(layout.shape[0]):
            for k in range(layout.shape[1]):
                layout_number = int(layout[i,k])
                if layout_number > 0:
                    field_im = all_images[layout_number - 1]
                    if field_im is None:
                        im_resized = np.zeros(target_field_resolution, dtype=np.uint8)
                    else:
                        im_resized = cv2.resize(field_im, target_field_resolution[:2], interpolation=cv2.INTER_CUBIC)
                    left_index = k * target_field_resolution[1]
                    right_index = (k+1) * target_field_resolution[1]
                    top_index = i * target_field_resolution[0]
                    bottom_index = (i+1) * target_field_resolution[0]
                    out_im[top_index:bottom_index,left_index:right_index,:] = im_resized#cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
        return out_im
        
def get_objects(table, row, col, field, row_id='Metadata_row', col_id='Metadata_column', field_id='Metadata_field'):
    return table[(table[row_id] == row) & (table[col_id] == col) & (table[field_id] == field)]

def get_objects_from_lists(table, rows, cols, fields, row_id='Metadata_row', col_id='Metadata_column', field_id='Metadata_field'):
    matrices = []
    for i in range(len(rows)):
        subdf = get_objects(table, rows[i], cols[i], fields[i], row_id=row_id, col_id=col_id, field_id=field_id)
        if (subdf.shape[0] >= 1):
            matrices.append(subdf.values)
    all_data = np.concatenate(matrices, axis=0)
    out = pd.DataFrame(data=all_data, columns=table.columns)
    return out

def plot_distributions(table, parameters):
    data = [table[f].values.reshape(-1,1) for f in parameters]
    fig, ax = plt.subplots(ncolumns=len(parameters), figsize=(6,len(parameters)*5))
    for i in range(len(data)):
        ax[i].hist(data[i], bins=100)
    fig.show()
    
def get_objects_based_on_constraints(table, kv_constraint_pairs):
    tmp_table = table
    for key, gle, value in kv_constraint_pairs:
        if gle == 'g':
            tmp_table = tmp_table[tmp_table[key] > value]
        if gle == 'l':
            tmp_table = tmp_table[tmp_table[key] < value]
        if gle == 'e':
            tmp_table = tmp_table[tmp_table[key] == value]
    return tmp_table

def print_scores(scores, name):
    print('Scores for {} are'.format(name))
    print(scores)
    print('Accuracy: {} (+/- {})'.format(np.mean(scores), 2*np.std(scores)))

def get_objects_list_constraints(table, nested_pairs):
    subtables = []
    for pairlist in nested_pairs:
        tmp = get_objects_based_on_constraints(table, pairlist)
        if (tmp.shape[0] >= 1):
            subtables.append(tmp.values)
    data = np.concatenate(subtables, axis=0)
    out = pd.DataFrame(data=data, columns=table.columns)
    return out

def show_scatter_data(scdata, labels=None):
    color_list = []
    colors = ['blue', 'red', 'green', 'pink', 'purple', 'yellow', 'orange', 'grey']
    if not (labels is None):
        for lb in labels:
            color_list.append(colors[lb])
        fig, ax = plt.subplots(figsize=(12,12))
        ax.scatter(scdata[:,0], scdata[:,1], color=color_list, alpha=0.1)
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.scatter(scdata[:,0], scdata[:,1])
        fig.show()
        
        
def visualize_cell_by_property(property_boolean, pos_index, visualizer_here=None):
    ### get global index of positive element
    if np.sum(property_boolean) <= pos_index:
        print('index of {} NOT valid for {} positive entries'.format(pos_index, np.sum(property_boolean)))
        return
    else:
        print('index of {} is valid for {} positive entries'.format(pos_index, np.sum(property_boolean)))
    tmp_index = 0
    for i in range(property_boolean.shape[0]):
        if property_boolean[i]:
            if tmp_index == pos_index:
                target_index = i
            tmp_index += 1
    visualizer_here.show_single_cell(target_index)
    return


def get_single_cell_images(property_boolean, num_images, visualizer_here=None, return_xy=False):
    if np.sum(property_boolean) == 0:
        images = []
        for _ in range(num_images):
            images.append((np.ones((71,71,3)) * 255).astype(np.uint8))
    else:
        indices = np.nonzero(property_boolean.flatten())[0].flatten()
        np.random.shuffle(indices)
        if not return_xy:
            images = [visualizer_here.visualize_single_cell(index, return_xy=return_xy) for index in indices[:num_images]]
        else:
            images = []
            yx = []
            for index in indices[:num_images]:
                im, y_val, x_val = visualizer_here.visualize_single_cell(index, return_xy=return_xy)
                images.append(im)
                yx.append((y_val, x_val))
        if len(images) < num_images:
            for _ in range(len(images), num_images):
                images.append((np.ones((71,71,3)) * 255).astype(np.uint8))
                if return_xy:
                    yx.append((0,0))
    if return_xy:
        return images, yx
    else:
        return images

def visualize_cells_in_grid(property_boolean, visualizer_here=None, max_grid_size=20, savename=None):
    images, yx = get_single_cell_images(property_boolean, max_grid_size**2, visualizer_here=visualizer_here, return_xy=True)
    fig, ax = plt.subplots(nrows=max_grid_size, ncols=max_grid_size, figsize=(max_grid_size*4, max_grid_size*4))
    im_index = 0
    for i in range(max_grid_size):
        for k in range(max_grid_size):
            ax[i,k].imshow(images[im_index])
            ax[i,k].set_title('Y: {} X: {}'.format(round(yx[im_index][0], 2), round(yx[im_index][1], 2)))
            im_index+=1
    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)