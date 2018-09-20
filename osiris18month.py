'''Welcome. This is the same file as tagcams.py, except it is written to be used with the
files within the LaunchPlus18Month_PSFImages folder. Good luck!'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from astropy.io import fits
from skimage import transform
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.backends.backend_pdf import PdfPages
import os
import glob

plt.ion()
plt.style.use('bmh')
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Myriad Pro'

def python_to_imageJ(list_of_coordinates_as_tuples): # takes in an array of data, spits out the coordinate of the maximum
    #python_coordinates =  np.unravel_index(array_data.argmax(), array_data.shape)
    #imageJ_coordinates = (python_coordinates[1], array_data.shape[0] - 1 - python_coordinates[0])
    array_shape = (2004, 2752)
    imageJ_coordinates = []
    for i in np.arange(len(list_of_coordinates_as_tuples)):
        imageJ_x = list_of_coordinates_as_tuples[i][1]
        imageJ_y = array_shape[0] - list_of_coordinates_as_tuples[i][0] - 1
        imageJ_coordinates.append((imageJ_x, imageJ_y))
    return imageJ_coordinates

def max_coordinates(name, print_on=False):
    '''if month == 18:
        folder = 'LaunchPlus18Month_PSFImages/'
    elif month == 6:
        folder = 'LaunchPlus6Month_PSFImages/'
    else:
        return 'Error! Month not recognized in the mission!'''
    data = fits.getdata(name)
    python_maxes = np.where(data == data.max())
    maxes_as_tuples = []
    for i in np.arange(len(python_maxes[0])):
        maxes_as_tuples.append((python_maxes[0][i], python_maxes[1][i]))

    if print_on:
        print ('Python coordinates (x,y): ', [(tup[1], tup[0]) for tup in maxes_as_tuples] )
    return maxes_as_tuples

def dark_correction(filename):
    data = fits.getdata(filename)
    active_indices = [54,-6, 140,-20]
    active_image = np.array(data[active_indices[0]:active_indices[1], active_indices[2]:active_indices[3]], dtype=float)
    # create an array that looks like ([1,2,3,4],[5,6,7,8]) to mark which pixel belongs to which offset
    offset_array = np.tile(np.arange(1,9).reshape(-1,4), (int(24/2),int(2608/4))) #(int(2004/2),int(2752/4)))

    # row and column coordinates for each dark pixel region
    # however, I think I need only region 4, which will be called dark_rows and dark_columns
    region0_rows, region0_cols = [(2,10), (2,10)], [(0,134), (-10, 2752)]
    region3_rows, region3_cols = [(0,2), (10,2004)], [(16,96), (16,96)]
    dark_rows, dark_columns = (10,34), (134,-10)

    darkpixels = data[dark_rows[0]:dark_rows[1], dark_columns[0]:dark_columns[1]]
    #dark_array = np.zeros((24,2608)) # the region 4 array
    mean_darkpixel_DN_list = []
    for offset_index in np.arange(1,9): # offsets range from 1 to 8
        counter = offset_index - 1
        mask_offset = offset_array == offset_index # boolean array based on offset values (1-8)
        mean_darkpixel_DN = np.mean(darkpixels[mask_offset]) # takes average DN of dark pixels belonging to unique offset value
        mean_darkpixel_DN_list.append(mean_darkpixel_DN)
        if counter < 4:
            active_image[0::2, counter::4] -= mean_darkpixel_DN
        elif counter >= 4:
            active_image[1::2, counter-4::4] -= mean_darkpixel_DN
        else:
            return 'Error in the counter'

    return mean_darkpixel_DN_list, active_image

def mass_darkcorrection():
    for filename in glob.glob('LaunchPlus18Month_PSFImages/*'):
        hdr = fits.getheader(filename)
        z = np.zeros((2004, 2752)) # creates array so everything but active image is zero
        dark_corrected_active_data = dark_correction(filename)[1]
        z[54:-6, 140:-20] = dark_corrected_active_data # fills in active image array in appropriate spot
        fits.writeto('18Month darkcorrected data/active {}'.format(filename[28:]), z, hdr)

# Generates list of dark-corrected images that have exposure times of 5s and 0.5s respectively
five_second_images = [img for img in glob.glob('18Month darkcorrected data/*') if np.round(fits.getheader(img)['EXPTIME'],2)==5]
half_second_images = [img for img in glob.glob('18Month darkcorrected data/*') if np.round(fits.getheader(img)['EXPTIME'],2)==0.5]
globs = glob.glob('18Month darkcorrected data/*')

def closest_point(coordinates, RA_DEC_location):
    '''Input: Array of maximum coordinates as tuples and a specified RA and DEC
    Finds the closest maximum coordinate to the specified RA and DEC star location'''
    assert len(RA_DEC_location) == 2
    squared_distances = np.sum((np.array(coordinates) - np.array(RA_DEC_location[::-1]))**2, axis=1)
    closest_coordinate = coordinates[squared_distances.argmin()]
    print ('Closest Coordinate in Python (row, column): ')
    return closest_coordinate

def plotterman(folder_index): # helper function I wrote in the command prompt to plot both photos
    plt.figure(1, figsize=(10,8))
    plt.imshow(fits.getdata(five_second_images[folder_index]),vmin=100,vmax=3500,cmap='gray')
    plt.title('5s image')
    plt.grid(False)
    plt.tight_layout()

    plt.figure(2, figsize=(10,8))
    plt.imshow(fits.getdata(half_second_images[folder_index]),vmin=100,vmax=500,cmap='gray')
    plt.title('0.5s image')
    plt.grid(False)
    plt.tight_layout()

def coordinates_to_tuples(bright_pixels):
    '''Make bright_pixels a list an array of the strings of the brightest pixels
    input should be in format ['(21, 35)', '28, 32', etc.]
    Returns: list of the same coordinates but in tuple format'''
    coordinates_list = []
    for word in bright_pixels:
        # adds a space for the coordinates that have no space in between them
        if word[word.find(',')+1] != ' ':
            word = ' '.join([word[:word.find(',')+1], word[1+word.find(','):]])
        python_row = float(word[word.find('(')+1 : word.find(',')])
        python_column = float(word[word.find(',')+2 : word.find(')')])
        coordinates_list.append( (python_row, python_column) )
    return coordinates_list

def make_excel(output_name):
    # gets the raw brightest pixels in the 0.5s and 5s exposure images
    trial1_brightestpixels_fiveseconds = np.array([max_coordinates(filename) for filename in five_second_images])
    trial1_brightestpixels_halfsecond = np.array([max_coordinates(filename) for filename in half_second_images])
    trial1_brightestpixels_halfsecond = [tuple(coord[0]) for coord in trial1_brightestpixels_halfsecond]

    truncated_five = [filename[27:] for filename in five_second_images]
    truncated_half = [filename[27:] for filename in half_second_images]

    fivesec_main_folder_location = [globs.index(filename) for filename in five_second_images]
    halfsec_main_folder_location = [globs.index(filename) for filename in half_second_images]
    assert (np.array(fivesec_main_folder_location) != np.array(halfsec_main_folder_location)).any()

    d_pandas = list(zip(truncated_five, trial1_brightestpixels_fiveseconds, fivesec_main_folder_location, \
    truncated_half, trial1_brightestpixels_halfsecond, halfsec_main_folder_location))

    df = pd.DataFrame(d_pandas, columns = ['Filename (5s)', '5s Brightest Pixel', 'Index in Folder (5s)', 'Filename (0.5s)', '0.5s Brightest Pixel', 'Index in Folder (5s)'])

    writer = pd.ExcelWriter(output_name)
    df.to_excel(writer)

    # formats the excel file to be prettier
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '#,##0.00', 'align':'center_across'})
    format1.set_align('center')
    worksheet.set_column('B:B', 18, format1)

    writer.save()

    return df

brights_5s = coordinates_to_tuples(pd.read_excel('LaunchPlus18Month_PSFImages Brightest Pixels.xlsx')['5s Brightest Pixel'].values)
brights_halfsec = coordinates_to_tuples(pd.read_excel('LaunchPlus18Month_PSFImages Brightest Pixels.xlsx')['0.5s Brightest Pixel'].values)

def brightest_star_subwindow(number_rows, number_columns, number_of_subplots_per_figure, num_figures, vmin_vmax=[180,2500], break_on=False):
    '''Plots the subwindow of the brightest star in each 5s exposure image.
    Input: number of rows for each figure, number of columns for each figure, minimum and maximum DNs (as a list)
    Output: Bunch of figures that plot the brightest star in a small subwindow.'''
    #assert number_rows * number_columns > number_of_subplots_per_figure

    brightest_pixels = brights_5s
    figure_size = (14,9)
    figures = []
    #fig1, fig2, fig3 = plt.figure(figsize=figure_size), plt.figure(figsize=figure_size), plt.figure(figsize=figure_size)
    #fig4, fig5, fig6 = plt.figure(figsize=figure_size), plt.figure(figsize=figure_size), plt.figure(figsize=figure_size)
    for fig_number in np.arange(num_figures):
        fig = plt.figure(figsize=figure_size)
        figures.append(fig)
    half_window_range = 5 # if you want 8x8 window for the stars, set to 4
    current_figure = 0

    #print (max_coordinates(filename), filename[27:])

    subplot_number = 1
    i = 0 # index for the image names
    for fig in figures:
        while subplot_number <= number_of_subplots_per_figure:
            filename = five_second_images[i]
            data = fits.getdata(filename)

            # plots data onto each subplot in a figure
            ax = figures[current_figure].add_subplot(number_rows, number_columns, subplot_number)
            img = ax.imshow(data, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='gray', aspect='equal')

            # create a square window centered on the brighest pixel in the image based on half_window_range
            ax.set_xlim(brightest_pixels[i][1]-half_window_range, brightest_pixels[i][1]+half_window_range)
            ax.set_ylim(brightest_pixels[i][0]-half_window_range, brightest_pixels[i][0]+half_window_range)
            ax.set_title(filename[34:], fontsize=9)
            ax.tick_params(labelsize = 9)

            subplot_number += 1
            i += 1
            assert subplot_number <= number_of_subplots_per_figure + 1
        else: # move onto next figure when counter is above number_of_subplots_per_figure
            figures[current_figure].tight_layout()
            if break_on:
                break
            current_figure += 1
            assert current_figure <= len(figures)

            subplot_number = 1

    return figures

def horizontal_vertical_PSF(filename, exposure=0.5, vmin_vmax=[180,2500], half_window_range=5, plot=False, lines=True): # plots intensity vs. pixel value of the horizontal and vertical profile
    full_data = fits.getdata(filename)
    if exposure == 0.5:
        file_index = half_second_images.index(filename)
        assert half_second_images[file_index][34:53].replace('S','')==fits.getheader(filename)['DATE_OBS'].replace('-','').replace(':','').replace('.',''), \
        'File is not in five_second_exposures or doesn\'t match'
        brightest_row, brightest_column = brights_halfsec[file_index]
        assert np.round(fits.getheader(half_second_images[file_index])['EXPTIME'],2) == exposure
    elif exposure == 5:
        file_index = five_second_images.index(filename)
        assert five_second_images[file_index][34:53].replace('S','')==fits.getheader(filename)['DATE_OBS'].replace('-','').replace(':','').replace('.',''), \
        'File is not in five_second_exposures or doesn\'t match'
        brightest_row, brightest_column = brights_5s[file_index]
        assert np.round(fits.getheader(five_second_images[file_index])['EXPTIME'],2) == exposure
    else:
        return 'No match for that exposure'

    # for even numbers
    if half_window_range % 2 == 0:
        row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range)
        col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range)
        xfront, xback = col_pixelrange[0], col_pixelrange[1]
        yback, yfront = row_pixelrange[1], row_pixelrange[0]
    # for odd numbers
    elif half_window_range % 2 == 1:
        row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range + 1)
        col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range + 1)
        xfront, xback = int(col_pixelrange[0]), int(col_pixelrange[1]-1)
        yback, yfront = int(row_pixelrange[1]-1), int(row_pixelrange[0])

    x, y = np.arange(int(col_pixelrange[0]),int(col_pixelrange[1])), np.arange(int(row_pixelrange[0]),int(row_pixelrange[1]))
    horizontal_PSF = full_data[int(brightest_row), int(col_pixelrange[0]):int(col_pixelrange[1])]
    vertical_PSF = full_data[int(row_pixelrange[0]):int(row_pixelrange[1]), int(brightest_column)]

    if plot:
        full_img = fits.getdata(filename)
        fig = plt.figure(figsize=(9,8))
        ax_image, ax_subwindow = fig.add_subplot(2,2,1), fig.add_subplot(2,2,2)
        ax_horizontal, ax_vertical = fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)

        title_pad = 14
        # plot the star subwindow
        rectangle = patches.Rectangle((brightest_column-65,brightest_row-50), 130,100, edgecolor='r',linewidth=1,facecolor='none')
        ax_image.add_patch(rectangle)
        if exposure == 0.5:
            ax_image.imshow(full_img,vmin=vmin_vmax[0]-180, vmax=vmin_vmax[1]-2200, cmap='gray', aspect='equal')
        elif exposure == 5:
            ax_image.imshow(full_img,vmin=vmin_vmax[0]-80, vmax=vmin_vmax[1], cmap='gray', aspect='equal')
        ax_image.set_xlabel('Pixel (x)')
        ax_image.set_ylabel('Pixel (y)')
        ax_image.set_title(filename[27:],pad=title_pad)
        ax_image.grid(False)
        #ax_image.axis('off')

        # plot the zoomed in image
        if exposure == 0.5:
            ax_subwindow.imshow(full_img,vmin=vmin_vmax[0]-180, vmax=vmin_vmax[1]-1500, cmap='gray', aspect='equal')
        elif exposure == 5:
            ax_subwindow.imshow(full_img,vmin=vmin_vmax[0], vmax=vmin_vmax[1]+1000, cmap='gray', aspect='equal')

        # determines whether or not the horizontal and vertical red lines trace over the center pixel
        if lines:
            ax_subwindow.axvline(brightest_column,color='r',linestyle='--', linewidth=0.5)
            ax_subwindow.axhline(brightest_row,color='r',linestyle='--', linewidth=0.5)

        ax_subwindow.set_ylim(yback, yfront) # inverted so that origin is upper left
        ax_subwindow.set_xlim(xfront, xback)
        ax_subwindow.set_title('Subwindow')
        ax_subwindow.grid(False)

        # plot the PSF on the x-axis and y-axis
        ax_horizontal.plot(x, horizontal_PSF)
        ax_horizontal.set_title('Horizontal PSF')
        ax_horizontal.set_xlabel('Pixel (x)')
        ax_horizontal.set_ylabel('DN')
        if exposure == 0.5:
            ax_horizontal.set_ylim(top=2400)

        ax_vertical.plot(y, vertical_PSF)
        ax_vertical.set_title('Vertical PSF')
        ax_vertical.set_xlabel('Pixel (y)')
        ax_vertical.set_ylabel('DN')
        if exposure == 0.5:
            ax_vertical.set_ylim(top=2400)

        fig.tight_layout()
        #fig.subplots_adjust(right=0.97, top=0.93, wspace=0.27)
        return np.array([x, horizontal_PSF]), np.array([y, vertical_PSF]), (fig, ax_image, ax_subwindow, ax_horizontal, ax_vertical), [xfront, xback, yfront, yback]

    else:
        return np.array([x, horizontal_PSF]), np.array([y, vertical_PSF]), [full_data,row_pixelrange,col_pixelrange], [xfront, xback, yfront, yback]

def diagonal_PSFs(filename, plot=False): # plots intensity vs. pixel value of the DIAGONAL profiles
    from skimage.draw import line # takes in (r0,c0,r1,c1) representing starting row/column and ending row/column and returns indices

    if filename in half_second_images:
        if plot == True:
            (fig, ax_image, ax_subwindow, ax_updiagonal, ax_downdiagonal), \
            [xfront, xback, yfront, yback] = horizontal_vertical_PSF(filename, plot=True, lines=False)[2:]
        elif plot == False:
            [xfront, xback, yfront, yback] = horizontal_vertical_PSF(filename, exposure=0.5, plot=False)[3]

    elif filename in five_second_images:
        if plot == True:
            (fig, ax_image, ax_subwindow, ax_updiagonal, ax_downdiagonal), \
            [xfront, xback, yfront, yback] = horizontal_vertical_PSF(filename, exposure=5, plot=True, lines=False)[2:]
        elif plot == False:
            [xfront, xback, yfront, yback] = horizontal_vertical_PSF(filename, exposure=5, plot=False)[3]

    data = fits.getdata(filename)
    # generates indices of diagonal lines
    # remember that x and y refer to column/row, not row/column and thus must be inverted
    up_diagonal_indices = line(yback, xfront, yfront ,xback)
    down_diagonal_indices = line(yfront, xfront, yback, xback)

    x = np.arange(len(down_diagonal_indices[0]))-np.median(np.arange(len(down_diagonal_indices[0])))
    ascending_data = data[up_diagonal_indices]
    descending_data = data[down_diagonal_indices]

    assert len(down_diagonal_indices[0]) == len(up_diagonal_indices[0]) and len(down_diagonal_indices[1]) == len(up_diagonal_indices[1]), \
    'Diagonal indices do not have the same length'

    if plot:
        # plots the diagonal lines over the subwindow image through the center (brightest) pixel
        ax_subwindow.plot(up_diagonal_indices[1], up_diagonal_indices[0], down_diagonal_indices[1], down_diagonal_indices[0], \
        color='r', linestyle='--', linewidth=0.5)

        # clears the axes for the up_diagonal and down_diagonal PSFs
        ax_downdiagonal.clear(), ax_updiagonal.clear()

        ax_updiagonal.plot(x, ascending_data)
        ax_updiagonal.set(xlabel='Distance from Center',ylabel='DN', title='Ascending Diagonal PSF')

        ax_downdiagonal.plot(x, descending_data)
        ax_downdiagonal.set(xlabel='Distance from Center',ylabel='DN', title='Descending Diagonal PSF')

        if filename in half_second_images:
            ax_downdiagonal.set_ylim(top=2400), ax_updiagonal.set_ylim(top=2400)
        elif filename in five_second_images:
            ax_downdiagonal.set_ylim(top=4200), ax_updiagonal.set_ylim(top=4200)
        else:
            return 'File is not in half_second_images or five_second_images'

        return fig, [x, ascending_data], [x, descending_data], [up_diagonal_indices, down_diagonal_indices]

    elif plot == False:
        return [x,ascending_data], [x, descending_data], [up_diagonal_indices, down_diagonal_indices]

# images that have stray light preventing the PSF from reaching close to zero
washed_out_images = np.array([41, 42])

def PSF_profiles_to_pdf(output_name, function='horizontal_vertical',half_window_range=5, exposure=0.5,dpi=800):

    plt.ioff() # so 77 figures don't pop up and close

    with PdfPages(output_name) as pdf:
        if exposure == 0.5:
            image_set = [img for img in half_second_images if img not in np.array(half_second_images)[washed_out_images]]
            #image_set = [img for img in half_second_images]
        elif exposure == 5:
            image_set = [img for img in five_second_images if img not in np.array(five_second_images)[washed_out_images]]
            #image_set = [img for img in five_second_images]
        i = 1
        for filename in image_set:
            if function == 'horizontal_vertical':
                active_figure = horizontal_vertical_PSF(filename,exposure=exposure,half_window_range=half_window_range,plot=True)[2][0]
            #filenumber_allfiles = globs.index(filename)
            elif function == 'diagonal_PSFs':
                active_figure = diagonal_PSFs(filename,plot=True)[0]
            else:
                return 'Invalid function'

            pdf.savefig(active_figure, dpi=dpi)
            #pdf.attach_note('In filelist containing all files, this is number ' + str(filenumber_allfiles))
            plt.close()
            print ('{} complete'.format(i))
            i += 1

    plt.ion()

#PSF_profiles_to_pdf('0.5s Diagonal PSFs.pdf',function = 'diagonal_PSFs',exposure=0.5)

def normalize_exposures(filename, cross_section='horizontal_vertical', half_window_range=5):
    '''Takes in a filename and returns the normalized PSF values based on exposure time
    Input: full filename as string
    Output: normalized PSF values in units of DN per second'''

    exposure = np.round(fits.getheader(filename)['EXPTIME'], 2)
    assert np.round(exposure,2) == 0.5 or np.round(exposure,2) == 5, 'Image exposure time is not 0.5s or 5s.'

    # Load the PSF data for the profiles
    if cross_section == 'horizontal_vertical':
        [x1,data1] = horizontal_vertical_PSF(filename, exposure=exposure, half_window_range=half_window_range, plot=False)[0]
        [x2,data2] = horizontal_vertical_PSF(filename, exposure=exposure, half_window_range=half_window_range, plot=False)[1]
    elif cross_section == 'diagonal':
        [x1,data1] = diagonal_PSFs(filename,plot=False)[0]
        [x2,data2] = diagonal_PSFs(filename,plot=False)[1]

    data1 /= exposure
    data2 /= exposure

    return [x1, data1], [x2, data2]

def check_normalized_maxima(filename,half_window_range=5):
    '''Checks whether all the peaks are at the center of the arrays'''
    exposure = fits.getheader(filename)['EXPTIME']
    assert np.round(exposure,2) == 0.5 or np.round(exposure,2) == 5, 'Image exposure time is not 0.5s or 5s.'

    normalized_horizontal, normalized_vertical = normalize_exposures(filename, 'diagonal', half_window_range=half_window_range)
    peak_indices = np.array([np.argmax(normalized_horizontal[1]), np.argmax(normalized_vertical[1])])
    return (peak_indices == half_window_range).all()

def plot_normalized_exposures_horizontalvertical(file_index,half_window_range=5, title_fontsize=12, axis_fontsize=10):
    assert file_index <= 77, 'File index is out of range'
    vmin_vmax = [180,2500]

    # all this stuff is for the half second exposure!!!
    half_exposure_data, row_pixelrange_half, col_pixelrange_half = horizontal_vertical_PSF(half_second_images[file_index],half_window_range=half_window_range)[2]
    five_exposure_data, row_pixelrange_five, col_pixelrange_five = horizontal_vertical_PSF(five_second_images[file_index],half_window_range=half_window_range,exposure=5)[2]
    fat_fig, [(ax_halfsec_subwindow, ax_5s_subwindow), (ax_halfsec_horizontal, ax_5s_horizontal), (ax_halfsec_vertical, ax_5s_vertical)] = plt.subplots(3,2, figsize=(11,9))
    # copied from the axis plotting code in horizontal_vertical_PSF()
    [xfront_halfsec, xback_halfsec, yfront_halfsec, yback_halfsec] = horizontal_vertical_PSF(half_second_images[file_index],half_window_range=half_window_range,exposure=0.5)[3]
    [xfront_5sec, xback_5sec, yfront_5sec, yback_5sec] = horizontal_vertical_PSF(five_second_images[file_index],half_window_range=half_window_range,exposure=5)[3]

    # plot the 0.5s subwindow
    ax_halfsec_subwindow.imshow(half_exposure_data,vmin=vmin_vmax[0]-180, vmax=vmin_vmax[1]-1500, cmap='gray', aspect='equal')
    ax_halfsec_subwindow.set_ylim(yback_halfsec, yfront_halfsec) # inverted so that origin is upper left
    ax_halfsec_subwindow.set_xlim(xfront_halfsec, xback_halfsec)
    ax_halfsec_subwindow.set_title(str(half_second_images[file_index][27:]), fontsize=title_fontsize, pad=14)
    ax_halfsec_subwindow.axvline(np.average([xfront_halfsec, xback_halfsec]),color='r',linestyle='--', linewidth=0.5)
    ax_halfsec_subwindow.axhline(np.average([yback_halfsec, yfront_halfsec]),color='r',linestyle='--', linewidth=0.5)
    ax_halfsec_subwindow.grid(False)

    # now plot the 5s subwindow
    ax_5s_subwindow.imshow(five_exposure_data,vmin=vmin_vmax[0], vmax=vmin_vmax[1]+1000, cmap='gray', aspect='equal')
    ax_5s_subwindow.set_ylim(yback_5sec, yfront_5sec) # inverted so that origin is upper left
    ax_5s_subwindow.set_xlim(xfront_5sec, xback_5sec)
    ax_5s_subwindow.set_title(str(five_second_images[file_index][27:]), fontsize=title_fontsize, pad=14)
    ax_5s_subwindow.axvline(np.average([xfront_5sec, xback_5sec]),color='r',linestyle='--', linewidth=0.5)
    ax_5s_subwindow.axhline(np.average([yback_5sec, yfront_5sec]),color='r',linestyle='--', linewidth=0.5)
    ax_5s_subwindow.grid(False)

    # now do the plotting of the normalized horizontal PSFs side-by-side
    halfsecond_normalized_horizontal, halfsecond_normalized_vertical =  normalize_exposures(half_second_images[file_index])
    ax_halfsec_horizontal.plot(halfsecond_normalized_horizontal[0], halfsecond_normalized_horizontal[1])
    ax_halfsec_horizontal.set_title('0.5s Exposure Normalized Horizontal PSF', fontsize=title_fontsize)
    ax_halfsec_horizontal.set_xlabel('Pixel (x)', fontsize=axis_fontsize)
    ax_halfsec_horizontal.set_ylabel('DN/sec', fontsize=axis_fontsize)
    ax_halfsec_horizontal.set_ylim(top=5200)

    fivesecond_normalized_horizontal, fivesecond_normalized_vertical =  normalize_exposures(five_second_images[file_index])
    ax_5s_horizontal.plot(fivesecond_normalized_vertical[0], fivesecond_normalized_vertical[1])
    ax_5s_horizontal.set_title('5s Exposure Normalized Horizontal PSF', fontsize=title_fontsize)
    ax_5s_horizontal.set_xlabel('Pixel (x)', fontsize=axis_fontsize)
    ax_5s_horizontal.set_ylabel('DN/sec', fontsize=axis_fontsize)
    ax_5s_horizontal.set_ylim(top=850) # the normalized values for 5s cap out at around 800

    # vertical data normalized PSF plotting for 0.5s and 5s
    ax_halfsec_vertical.plot(halfsecond_normalized_vertical[0], halfsecond_normalized_vertical[1])
    ax_halfsec_vertical.set_title('0.5s Exposure Normalized Vertical PSF', fontsize=title_fontsize)
    ax_halfsec_vertical.set_xlabel('Pixel (y)', fontsize=axis_fontsize)
    ax_halfsec_vertical.set_ylabel('DN/sec', fontsize=axis_fontsize)
    ax_halfsec_vertical.set_ylim(top=5200)

    ax_5s_vertical.plot(fivesecond_normalized_vertical[0], fivesecond_normalized_vertical[1])
    ax_5s_vertical.set_title('5s Exposure Normalized Vertical PSF', fontsize=title_fontsize)
    ax_5s_vertical.set_xlabel('Pixel (y)', fontsize=axis_fontsize)
    ax_5s_vertical.set_ylabel('DN/sec', fontsize=axis_fontsize)
    ax_5s_vertical.set_ylim(top=850) # the normalized values for 5s cap out at around 800

    fat_fig.tight_layout()

    return fat_fig, (ax_halfsec_subwindow, ax_5s_subwindow,ax_halfsec_horizontal,ax_5s_horizontal,ax_halfsec_vertical,ax_5s_vertical)

def plot_normalized_diagonal_exposures(file_index,half_window_range=5, title_fontsize=12, axis_fontsize=10):
    assert file_index <= 77, 'File index is out of range'
    assert file_index not in washed_out_images
    vmin_vmax = [180,2500]

    full_img_half = fits.getdata(half_second_images[file_index])
    full_img_five = fits.getdata(five_second_images[file_index])

    fig, [(ax_image_half, ax_image_five), (ax_halfsec_subwindow, ax_fivesec_subwindow), \
    (ax_halfsec_diagonals, ax_fivesec_diagonals)] = plt.subplots(3,2, figsize=(11,9))

    # copied from the axis plotting code in horizontal_vertical_PSF()
    [xfront_halfsec, xback_halfsec, yfront_halfsec, yback_halfsec] = horizontal_vertical_PSF(half_second_images[file_index],half_window_range=half_window_range,exposure=0.5)[3]
    [xfront_fivesec, xback_fivesec, yfront_fivesec, yback_fivesec] = horizontal_vertical_PSF(five_second_images[file_index],half_window_range=half_window_range,exposure=5)[3]

    # get the diagonal lines from diagonal_PSFs()
    ascending_diagonal_halfsec, descending_diagonal_halfsec = diagonal_PSFs(half_second_images[file_index])[2]
    ascending_diagonal_fivesec, descending_diagonal_fivesec = diagonal_PSFs(five_second_images[file_index])[2]

    # all this is for the two images at the top!
    file_index_halfsecond = file_index
    file_index_fiveseconds = file_index

    brightest_row_half, brightest_column_half = brights_halfsec[file_index]
    brightest_row_five, brightest_column_five = brights_5s[file_index]

    assert np.round(fits.getheader(half_second_images[file_index])['EXPTIME'],2) == 0.5
    assert half_second_images[file_index_halfsecond][34:53].replace('S','')==fits.getheader(half_second_images[file_index_halfsecond])['DATE_OBS'].replace('-','').replace(':','').replace('.',''), \
    'File is not in five_second_exposures or doesn\'t match'
    assert five_second_images[file_index_fiveseconds][34:53].replace('S','')==fits.getheader(five_second_images[file_index_fiveseconds])['DATE_OBS'].replace('-','').replace(':','').replace('.',''), \
    'File is not in five_second_exposures or doesn\'t match'
    assert np.round(fits.getheader(five_second_images[file_index])['EXPTIME'],2) == 5

    title_pad = 14
    # plot the 0.5s star full image
    rectangle_halfsecond = patches.Rectangle((brightest_column_half-65,brightest_row_half-50), 130,100, edgecolor='r',linewidth=1,facecolor='none')

    ax_image_half.add_patch(rectangle_halfsecond)
    ax_image_half.imshow(full_img_half,vmin=vmin_vmax[0]-180, vmax=vmin_vmax[1]-2200, cmap='gray', aspect='equal')
    ax_image_half.set_xlabel('Pixel (x)')
    ax_image_half.set_ylabel('Pixel (y)')
    ax_image_half.set_title(half_second_images[file_index_halfsecond][27:],pad=title_pad)
    ax_image_half.grid(False)

    # plotting the 5s star full image
    rectangle_fivesecond = patches.Rectangle((brightest_column_five-65,brightest_row_five-50), 130,100, edgecolor='r',linewidth=1,facecolor='none')
    ax_image_five.add_patch(rectangle_fivesecond)
    ax_image_five.imshow(full_img_five,vmin=vmin_vmax[0]-80, vmax=vmin_vmax[1], cmap='gray', aspect='equal')
    ax_image_five.set_xlabel('Pixel (x)')
    ax_image_five.set_ylabel('Pixel (y)')
    ax_image_five.set_title(five_second_images[file_index_fiveseconds][27:],pad=title_pad)

    ax_image_five.grid(False)

    # The 0.5s subwindow stuff
    ax_halfsec_subwindow.imshow(fits.getdata(half_second_images[file_index]), vmin=vmin_vmax[0]-180, vmax=vmin_vmax[1]-1500, cmap='gray')
    ax_halfsec_subwindow.set_ylim(yback_halfsec, yfront_halfsec) # inverted so that origin is upper left
    ax_halfsec_subwindow.set_xlim(xfront_halfsec, xback_halfsec)
    ax_halfsec_subwindow.set_title('0.5s Subwindow', fontsize=title_fontsize, pad=10)
    ax_halfsec_subwindow.plot(ascending_diagonal_halfsec[1], ascending_diagonal_halfsec[0], linewidth=0.5, linestyle='--', c='red')
    ax_halfsec_subwindow.plot(descending_diagonal_halfsec[1], descending_diagonal_halfsec[0], linewidth=0.5, linestyle='--', c='red')
    ax_halfsec_subwindow.grid(False)

    # the 5s exposure subwindow along with the diagonal lines
    ax_fivesec_subwindow.imshow(fits.getdata(five_second_images[file_index]), vmin=vmin_vmax[0], vmax=vmin_vmax[1]+1000, cmap='gray')
    ax_fivesec_subwindow.set_ylim(yback_fivesec, yfront_fivesec) # inverted so that origin is upper left
    ax_fivesec_subwindow.set_xlim(xfront_fivesec, xback_fivesec)
    ax_fivesec_subwindow.set_title('5s Subwindow', fontsize=title_fontsize, pad=10)
    ax_fivesec_subwindow.plot(ascending_diagonal_fivesec[1], ascending_diagonal_fivesec[0], linewidth=0.5, linestyle='--', c='red')
    ax_fivesec_subwindow.plot(descending_diagonal_fivesec[1], descending_diagonal_fivesec[0], linewidth=0.5, linestyle='--', c='red')
    ax_fivesec_subwindow.grid(False)

    # plot normalized ascending diagonal PSFs side-by-side
    halfsecond_normalized_ascending, halfsecond_normalized_descending = normalize_exposures(half_second_images[file_index], 'diagonal')
    ax_halfsec_diagonals.plot(halfsecond_normalized_ascending[0], halfsecond_normalized_ascending[1], label='ascending')
    ax_halfsec_diagonals.plot(halfsecond_normalized_descending[0], halfsecond_normalized_descending[1], c='red', label='descending')
    ax_halfsec_diagonals.set_title('0.5s Exposure Normalized PSF', fontsize=title_fontsize)
    ax_halfsec_diagonals.set_xlabel('Distance from Center', fontsize=axis_fontsize)
    ax_halfsec_diagonals.set_ylabel('DN/sec', fontsize=axis_fontsize)
    ax_halfsec_diagonals.legend()
    ax_halfsec_diagonals.set_ylim(top=5200)

    # 5s exposure stuff plotted
    fivesecond_normalized_ascending, fivesecond_normalized_descending = normalize_exposures(five_second_images[file_index], 'diagonal')
    ax_fivesec_diagonals.plot(fivesecond_normalized_ascending[0], fivesecond_normalized_ascending[1], label='ascending')
    ax_fivesec_diagonals.plot(fivesecond_normalized_descending[0], fivesecond_normalized_descending[1], c='red', label='descending')
    ax_fivesec_diagonals.set_title('5s Exposure Normalized PSF', fontsize=title_fontsize)
    ax_fivesec_diagonals.set_xlabel('Distance from Center', fontsize=axis_fontsize)
    ax_fivesec_diagonals.set_ylabel('DN/sec', fontsize=axis_fontsize)
    ax_fivesec_diagonals.legend()
    ax_fivesec_diagonals.set_ylim(top=850) # the normalized values for 5s cap out at around 800

    fig.tight_layout()

    return fig

def PDF_normalized_PSFs(output_name, cross_section='horizontal_vertical', half_window_range=5, dpi=800):

    plt.ioff() # so 77 figures don't pop up and close
    i = 1
    with PdfPages(output_name) as pdf:
        for filenumber in [i for i in np.arange(len(five_second_images)) if i not in washed_out_images]:
            if cross_section == 'horizontal_vertical':
                active_figure = plot_normalized_exposures_horizontalvertical(filenumber,half_window_range)[0]
            elif cross_section == 'diagonal':
                active_figure = plot_normalized_diagonal_exposures(filenumber, half_window_range)
            pdf.savefig(active_figure, dpi=dpi)
            plt.close()
            print ('{} plots completed'.format(i))
            i += 1
    plt.ion()

#PDF_normalized_PSFs('18 Month PSFs/Normalized Horizontal-Vertical PSFs.pdf', 'horizontal_vertical')
#PDF_normalized_PSFs('Normalized Diagonal PSFs.pdf', 'diagonal')

def replace_peaks(filenumber, cross_section='horizontal_vertical', plot=False, threshold=780, vmin_vmax=[180,2500], half_window_range=5):
    assert filenumber not in washed_out_images, 'This image was marked to have stray light in it!'

    if cross_section == 'horizontal_vertical':
        horizontal_5sec, vertical_5sec = normalize_exposures(five_second_images[filenumber])
        horizontal_half_sec, vertical_half_sec = normalize_exposures(half_second_images[filenumber])
    elif cross_section == 'diagonal':
        '''NOTE: For the sake of not having to change all the terms in this function,
        I left these variable names untouched. However, these are NOT horizontal and
        vertical PSFs, but the ascending and descending diagonal PSFs.'''
        horizontal_5sec, vertical_5sec = normalize_exposures(five_second_images[filenumber], 'diagonal')
        horizontal_half_sec, vertical_half_sec = normalize_exposures(half_second_images[filenumber], 'diagonal')

    # tests whether the pixel coordinates line up
    # just found out that max pixel has already been set because we used 5s exposure max coordinates for the 0.5s exposure images too!
    assert horizontal_5sec[0].all()==horizontal_half_sec[0].all(), 'Pixel x-coordinates do not line up.'
    assert vertical_5sec[0].all()==vertical_half_sec[0].all(), 'Pixel y-coordinates do not line up.'

    # locates the indices that are above the threshold in the 5s exposure images
    above_threshold_horizontally =  np.where(horizontal_5sec[1] > threshold)[0]
    above_threshold_vertically = np.where(vertical_5sec[1] > threshold)[0]

    # replace those values with the corresponding values from the 0.5s exposure PSF
    horizontal_5sec[1][above_threshold_horizontally] = horizontal_half_sec[1][above_threshold_horizontally]
    vertical_5sec[1][above_threshold_vertically] = vertical_half_sec[1][above_threshold_vertically]

    # now plot that
    if plot:
        # all this for just the one image at the top
        import matplotlib.patches as patches
        full_img = fits.getdata(five_second_images[filenumber])
        brightest_row, brightest_column = brights_5s[filenumber]
        rectangle = patches.Rectangle((brightest_column-65,brightest_row-50), 130,100, edgecolor='r',linewidth=1,facecolor='none')

        # some good aesthetics settings
        title_fontsize = 10
        axis_fontsize = 9
        upper_limit = 5200
        labelpad = 10

        fig = plt.figure(figsize=(11,9))
        ax_image = fig.add_subplot(3,1,1)
        ax_horizontal_replaced, ax_vertical_replaced = fig.add_subplot(3,2,3), fig.add_subplot(3,2,4)
        ax_normalized_horizontal, ax_normalized_vertical = fig.add_subplot(3,2,5), fig.add_subplot(3,2,6)

        # plot the full image and the rectangle around the location
        ax_image.imshow(full_img,vmin=vmin_vmax[0]-80, vmax=vmin_vmax[1]-2200, cmap='gray', aspect='equal')
        ax_image.add_patch(rectangle)
        ax_image.set_title('0.5s image: {1} \n 5s image: {0}'.format(five_second_images[filenumber][27:],half_second_images[filenumber][27:]),fontsize=title_fontsize, pad=15)
        ax_image.set_xlabel('Pixel (x)',fontsize=axis_fontsize), ax_image.set_ylabel('Pixel (y)',fontsize=axis_fontsize)
        ax_image.grid(False)

        # plots the replacement data
        ax_horizontal_replaced.plot(horizontal_5sec[0],horizontal_5sec[1])
        ax_horizontal_replaced.plot(horizontal_5sec[0][above_threshold_horizontally],horizontal_5sec[1][above_threshold_horizontally],color='red', linestyle='none',marker='o')
        ax_horizontal_replaced.set_title('Replaced Horizontal PSF',fontsize=title_fontsize)
        ax_horizontal_replaced.set_ylabel('DN/sec',fontsize=axis_fontsize,labelpad=labelpad)
        ax_horizontal_replaced.set_xlabel('Pixel (x)',fontsize=axis_fontsize)
        ax_horizontal_replaced.set_ylim(top=upper_limit)

        ax_vertical_replaced.plot(vertical_5sec[0],vertical_5sec[1])
        ax_vertical_replaced.plot(vertical_5sec[0][above_threshold_vertically],vertical_5sec[1][above_threshold_vertically],color='red', linestyle='none', marker='o')
        ax_vertical_replaced.set_title('Replaced Vertical PSF',fontsize=title_fontsize)
        ax_vertical_replaced.set_ylabel('DN/sec',fontsize=axis_fontsize, labelpad=labelpad)
        ax_vertical_replaced.set_xlabel('Pixel (y)',fontsize=axis_fontsize)
        ax_vertical_replaced.set_ylim(top=upper_limit)

        # put the 0.5s and 5s PSFs on the same plot, but separated by horizontal and vertical PSF
        if cross_section == 'horizontal_vertical':
            halfsecond_normalized_horizontal, halfsecond_normalized_vertical =  normalize_exposures(half_second_images[filenumber])
            fivesecond_normalized_horizontal, fivesecond_normalized_vertical =  normalize_exposures(five_second_images[filenumber])
        elif cross_section == 'diagonal':
            '''Again, these terms are to avoid changing the rest of the variable names but these
            are the normalized data for the ascending and descending diagonals.'''
            halfsecond_normalized_horizontal, halfsecond_normalized_vertical =  normalize_exposures(half_second_images[filenumber], 'diagonal')
            fivesecond_normalized_horizontal, fivesecond_normalized_vertical =  normalize_exposures(five_second_images[filenumber], 'diagonal')

        ax_normalized_horizontal.plot(halfsecond_normalized_horizontal[0],halfsecond_normalized_horizontal[1],label='0.5s')
        ax_normalized_horizontal.plot(fivesecond_normalized_horizontal[0],fivesecond_normalized_horizontal[1],color='red',label='5s')
        ax_normalized_horizontal.legend()
        ax_normalized_horizontal.set_ylabel('DN/sec',fontsize=axis_fontsize,labelpad=labelpad)
        ax_normalized_horizontal.set_xlabel('Pixel (x)',fontsize=axis_fontsize)
        ax_normalized_horizontal.set_title('Normalized Horizontal PSF',fontsize=title_fontsize)
        ax_normalized_horizontal.set_ylim(top=upper_limit)

        ax_normalized_vertical.plot(halfsecond_normalized_vertical[0],halfsecond_normalized_vertical[1],label='0.5s')
        ax_normalized_vertical.plot(fivesecond_normalized_vertical[0],fivesecond_normalized_vertical[1],color='red',label='5s')
        ax_normalized_vertical.legend()
        ax_normalized_vertical.set_ylabel('DN/sec',fontsize=axis_fontsize,labelpad=labelpad)
        ax_normalized_vertical.set_xlabel('Pixel (y)',fontsize=axis_fontsize)
        ax_normalized_vertical.set_title('Normalized Vertical PSF',fontsize=title_fontsize)
        ax_normalized_vertical.set_ylim(top=upper_limit)

        # replacing the labels for the diagonal case
        ax_horizontal_replaced.set_title('Replaced Ascending PSF',fontsize=title_fontsize)
        ax_vertical_replaced.set_title('Replaced Descending PSF',fontsize=title_fontsize)
        ax_normalized_horizontal.set_title('Normalized Ascending PSFs',fontsize=title_fontsize)
        ax_normalized_vertical.set_title('Normalized Descending PSFs',fontsize=title_fontsize)

        ax_horizontal_replaced.set_xlabel('Distance from Center',fontsize=axis_fontsize)
        ax_vertical_replaced.set_xlabel('Distance from Center',fontsize=axis_fontsize)
        ax_normalized_horizontal.set_xlabel('Distance from Center',fontsize=axis_fontsize)
        ax_normalized_vertical.set_xlabel('Distance from Center',fontsize=axis_fontsize)

        fig.tight_layout()
        fig.subplots_adjust(wspace=.2)

        return horizontal_5sec, vertical_5sec, (ax_image,ax_horizontal_replaced,ax_vertical_replaced,ax_normalized_horizontal,ax_normalized_vertical), fig

    elif plot==False:
        return horizontal_5sec, vertical_5sec

def PDF_peak_remover(output_name, cross_section='horizontal_vertical', threshold=780, half_window_range=5, dpi=800):

    plt.ioff() # so 77 figures don't pop up and close

    with PdfPages(output_name) as pdf:
        i = 1
        for filenumber in [i for i in np.arange(len(five_second_images)) if i not in washed_out_images]:
            active_figure = replace_peaks(filenumber,cross_section,threshold=threshold,half_window_range=half_window_range,plot=True)[-1]
            print ('{} plot completed'.format(i))
            pdf.savefig(active_figure, dpi=dpi)
            plt.close()
            i += 1

    plt.ion()

#PDF_peak_remover('Horizontal-Vertical PSFs peaks removed.pdf',cross_section='horizontal_vertical')
#PDF_peak_remover('18 Month PSFs/2018 Diagonal PSFs peaks removed.pdf','diagonal', 770)

'''DOY100_5s_images = [img for img in five_second_images if img in globs[:58] and img not in np.array(five_second_images)[washed_out_images]]
DOY100_halfsec_images = [img for img in half_second_images if img in globs[:58] and img not in np.array(half_second_images)[washed_out_images]]
DOY103_ncm_5s_images = [img for img in five_second_images if img in globs[58:116] and img not in np.array(five_second_images)[washed_out_images]]
DOY103_ncm_halfsec_images = [img for img in half_second_images if img in globs[58:116] and img not in np.array(half_second_images)[washed_out_images]]
DOY103_nft_5s_images = [img for img in five_second_images if img in globs[116:] and img not in np.array(five_second_images)[washed_out_images]]
DOY103_nft_halfsec_images = [img for img in half_second_images if img in globs[116:] and img not in np.array(half_second_images)[washed_out_images]]

dict_coordinates_as_regions = {(0, 0): 'center', (0, -8): 'bottom', (0, 8): 'top', (11, 8): 'top left', (-11, 8): 'top right',
                               (-11, -8): 'bottom right', (11, -8): 'bottom left', (11, 0): 'left', (-11, 0): 'right',
                               (20, 15): 'corner top left', (20, -15): 'corner bottom left', (-20, -15): 'corner bottom right', (-20, 15): 'corner top right'}'''

def overlaid_PSFs(DOY=100,camera='nft',gain=8, cmap='tab10',linewidth=1.5):
    #from matplotlib import cm

    DOY100_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY100', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    DOY103_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY103', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    raw_pointingoffsets = np.concatenate((DOY100_pointingoffsets, DOY103_pointingoffsets))
    pointing_offset_coordinates = coordinates_to_tuples(['({})'.format(string) for string in raw_pointingoffsets if string != 'nan' and string != '---'])

    if DOY == 100:
        gain8_5s_images = [img for img in DOY100_5s_images if fits.getheader(img)['TCGAIN'] == 8]
        gain10_5s_images = [img for img in DOY100_5s_images if fits.getheader(img)['TCGAIN'] == 10]
        gain8_halfsec_images = [img for img in DOY100_halfsec_images if fits.getheader(img)['TCGAIN'] == 8]
        gain10_halfsec_images = [img for img in DOY100_halfsec_images if fits.getheader(img)['TCGAIN'] == 10]

    elif DOY==103 and camera=='ncm':
        gain8_5s_images = [img for img in DOY103_ncm_5s_images if fits.getheader(img)['TCGAIN'] == 8]
        gain10_5s_images = [img for img in DOY103_ncm_5s_images if fits.getheader(img)['TCGAIN'] == 10]
        gain8_halfsec_images = [img for img in DOY103_ncm_halfsec_images if fits.getheader(img)['TCGAIN'] == 8]
        gain10_halfsec_images = [img for img in DOY103_ncm_halfsec_images if fits.getheader(img)['TCGAIN'] == 10]

    elif DOY==103 and camera=='nft':
        gain8_5s_images = [img for img in DOY103_nft_5s_images if fits.getheader(img)['TCGAIN'] == 8]
        gain10_5s_images = [img for img in DOY103_nft_5s_images if fits.getheader(img)['TCGAIN'] == 10]
        gain8_halfsec_images = [img for img in DOY103_nft_halfsec_images if fits.getheader(img)['TCGAIN'] == 8]
        gain10_halfsec_images = [img for img in DOY103_nft_halfsec_images if fits.getheader(img)['TCGAIN'] == 10]

    assert (np.array([fits.getheader(file)['INSTRUME'].lower() for file in gain8_5s_images]) == camera).all(), 'Something wrong with the gain-8 5s files.'
    assert (np.array([fits.getheader(file)['INSTRUME'].lower() for file in gain10_5s_images]) == camera).all(), 'Something wrong with the gain-10 5s files.'
    assert (np.array([fits.getheader(file)['INSTRUME'].lower() for file in gain8_halfsec_images]) == camera).all(), 'Something wrong with the gain-8 0.5s files.'
    assert (np.array([fits.getheader(file)['INSTRUME'].lower() for file in gain10_halfsec_images]) == camera).all(), 'Something wrong with the gain-10 0.5s files.'
    assert gain == 8 or gain==10, 'Invalid gain value. We only accept 8 or 10, thank you very much'

    fig, (ax_horizontal, ax_vertical) = plt.subplots(2,1,figsize=(16,9),sharex=True,sharey=True)
    number_of_lines = len(gain8_5s_images)
    #color = np.linspace(0,1,number_of_lines)
    cm = plt.get_cmap(cmap)
    ax_horizontal.set_color_cycle([cm(1.*i/number_of_lines) for i in np.arange(number_of_lines,0,-1)])
    ax_vertical.set_color_cycle([cm(1.*i/number_of_lines) for i in np.arange(number_of_lines,0,-1)])


    if gain==8:
        #for color_index, file_5sec, file_halfsec in zip(color, gain8_5s_images, gain8_halfsec_images):
        for file_5sec, file_halfsec in zip(gain8_5s_images, gain8_halfsec_images):
            file_5sec_index, file_halfsec_index = five_second_images.index(file_5sec), half_second_images.index(file_halfsec)
            assert file_5sec_index == file_halfsec_index, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'

            # searches in all LaunchPlus6Month_PSFImages files for the corresponding index
            all_files_halfsecond_index, all_files_fivesecond_index =  globs.index(file_halfsec), globs.index(file_5sec)
            assert all_files_halfsecond_index < len(globs) and all_files_fivesecond_index < len(globs)
            assert globs[all_files_halfsecond_index] not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
            assert globs[all_files_fivesecond_index] not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'

            # first finds the pointing offset coordinates and then matches it to the region
            region_halfsecond = dict_coordinates_as_regions[pointing_offset_coordinates[all_files_halfsecond_index]]
            region_fivesecond = dict_coordinates_as_regions[pointing_offset_coordinates[all_files_fivesecond_index]]
            assert region_halfsecond == region_fivesecond, 'Regions are not the same!'

            replaced_horizontal, replaced_vertical = replace_peaks(file_5sec_index)
            normalized_horizontal = replaced_horizontal[1] / np.max(replaced_horizontal[1])
            normalized_vertical = replaced_vertical[1] / np.max(replaced_vertical[1])

            ax_horizontal.plot(normalized_horizontal, label=region_halfsecond, linewidth=linewidth)#, color=cm.Accent(color_index), linewidth=1.5)
            ax_vertical.plot(normalized_vertical, label=region_halfsecond, linewidth=linewidth)#, color=cm.Accent(color_index), linewidth=1.5)

    elif gain==10:
        #for color_index, file_5sec, file_halfsec in zip(color, gain10_5s_images, gain10_halfsec_images):
        for file_5sec, file_halfsec in zip(gain10_5s_images, gain10_halfsec_images):
            file_5sec_index, file_halfsec_index = five_second_images.index(file_5sec), half_second_images.index(file_halfsec)
            assert file_5sec_index == file_halfsec_index, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'

            # searches in all LaunchPlus6Month_PSFImages files for the corresponding index
            all_files_halfsecond_index, all_files_fivesecond_index =  globs.index(file_halfsec), globs.index(file_5sec)
            assert all_files_halfsecond_index < len(globs) and all_files_fivesecond_index < len(globs)
            assert globs[all_files_halfsecond_index] not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
            assert globs[all_files_fivesecond_index] not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'

            # first finds the pointing offset coordinates and then matches it to the region
            region_halfsecond = dict_coordinates_as_regions[pointing_offset_coordinates[all_files_halfsecond_index]]
            region_fivesecond = dict_coordinates_as_regions[pointing_offset_coordinates[all_files_fivesecond_index]]
            assert region_halfsecond == region_fivesecond, 'Regions are not the same!'

            replaced_horizontal, replaced_vertical = replace_peaks(file_5sec_index)
            normalized_horizontal = replaced_horizontal[1] / np.max(replaced_horizontal[1])
            normalized_vertical = replaced_vertical[1] / np.max(replaced_vertical[1])
            #style = np.random.choice(['-','--'],p=[0.7,0.3])
            style='-'
            ax_horizontal.plot(normalized_horizontal, linestyle=style,label=region_halfsecond, linewidth=linewidth)#, color=cm.Accent(color_index), linewidth=1.5)
            ax_vertical.plot(normalized_vertical, linestyle=style,label=region_halfsecond, linewidth=linewidth)#, color=cm.Accent(color_index), linewidth=1.5)

    ax_horizontal.legend(loc='best'), ax_vertical.legend(loc='best')
    ax_horizontal.set_ylabel('Normalized Horizontal PSF',labelpad=14)
    ax_vertical.set_ylabel('Normalized Vertical PSF', labelpad=14)
    #ax_horizontal.set_ylim(top=5000), ax_vertical.set_ylim(top=5000)

    fig.suptitle('Camera: {0}\n DOY: {1}     Gain: {2}'.format(camera.upper(), DOY, gain),fontsize=14)
    fig.tight_layout()
    #fig.subplots_adjust(left=.06, top=.9,wspace=.22)
    fig.subplots_adjust(left=.06,right=.98,top=.92,hspace=0)

    return replaced_horizontal, replaced_vertical

def assign_region(coordinate):
    '''Returns a region that the coordinate is in. Possible regions can be found in the dictionary region_colors.
    Note that this method has only been tested on the files in LaunchPlus18Month_PSFImages after checking their
    brightest pixel values, so it won't work for everything!'''

    row, column = coordinate

    if row < 250: # corner top left or corner top right
        if column < 500:
            return 'corner top left'
        elif column > 2400:
            return 'corner top right'
        else:
            'Could not match to corner top left or corner top right'
    elif row > 1700: # corner bottom left/right
        if column < 500:
            return 'corner bottom left'
        elif column > 2400:
            return 'corner bottom right'
        else:
            'Could not match to corner bottom left or corner bottom right'

    elif row in range(500, 1600): # gets the middle 9 regions
        if column in range(650,1100): # top left, left, and bottom left
            if row in range(500,700):
                return 'top left'
            elif row in range(900,1100):
                return 'left'
            elif row in range(1450,1600):
                return 'bottom left'
            else:
                'No match after narrowing down to the left side of the middle grid'

        elif column in range(1900,2200): # top right, right, and bottom right
            if row in range(500,700):
                return 'top right'
            elif row in range(900,1100):
                return 'right'
            elif row in range(1450,1600):
                return 'bottom right'
            else:
                'No match after narrowing down to the right side of the middle grid'

        elif column in range(1200, 1600): # now the gnarly stuff——the middle column
            if row in range(500,800): # top
                return 'top'
            elif row in range(1000, 1200): # center
                return 'center'
            elif row in range(1300, 1600): # bottom
                return 'bottom'
            else:
                return 'Narrowed down to the middle column but couldn\'t find a match'
        else:
            return 'Okay so it\'s not a corner piece but it doesn\'t fit the columns of the middle.'

    else:
        return 'This thing has no row to go!'

def separate_regions(diagonal_path='left_down', cross_section='horizontal_vertical',camera='NFT',gain=8, cmap='tab10',linewidth=1.5):
    # Plots the horizontal and vertical PSFs for points that follow the specified diagonal_path
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #DOY100_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY100', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #DOY103_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY103', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #raw_pointingoffsets = np.concatenate((DOY100_pointingoffsets, DOY103_pointingoffsets))
    #pointing_offset_coordinates = coordinates_to_tuples(['({})'.format(string) for string in raw_pointingoffsets if string != 'nan' and string != '---'])

    # specifies the paths taken to plot the diagonal PSFs
    diagonal_paths = {'left_down': ['corner top left', 'top left', 'center', 'bottom right', 'corner bottom right'],
                      'left_up': ['corner bottom left', 'bottom left', 'center', 'top right', 'corner top right'],
                      'central': ['top', 'left', 'bottom', 'right', 'center']}

    assert camera == 'NFT' or camera == 'NCM', 'Invalid camera!'
    assert gain == 8 or gain==10, 'Invalid gain value. We only accept 8 or 10, thank you very much'

    # loads in the images based on gain and camera values
    camera_halfsec = [img for img in half_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(half_second_images)[washed_out_images]]

    camera_fivesec = [img for img in five_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(five_second_images)[washed_out_images]]

    camera_halfsec_set1 = camera_halfsec[::2]
    camera_halfsec_set2 = camera_halfsec[1::2]
    camera_fivesec_set1 = camera_fivesec[::2]
    camera_fivesec_set2 = camera_fivesec[1::2]

    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_halfsec_set1]) == camera).all()
    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_halfsec_set2]) == camera).all()
    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_fivesec_set1]) == camera).all()
    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_fivesec_set2]) == camera).all()


    # for some reason there's two sets of images!!
    fig1, (ax_horizontal1, ax_vertical1) = plt.subplots(2,1,figsize=(16,9),sharex=True,sharey=True)
    fig2, (ax_horizontal2, ax_vertical2) = plt.subplots(2,1,figsize=(16,9),sharex=True,sharey=True)

    # specifies which color the box around a region shoudl be
    region_colors = {'center': 'tab:red', 'top left': 'orange', 'bottom left': 'orange',
                     'corner top right': 'tab:green', 'corner top left': 'tab:green',
                     'top right': 'tab:blue', 'bottom right': 'tab:blue',
                     'top': 'tab:green', 'bottom': 'tab:purple', 'left': 'orange', 'right':'tab:blue',
                     'corner bottom left': 'tab:purple', 'corner bottom right':'tab:purple'}

    # gets the colors for use in the legend
    colors1 = []
    colors2 = []
    rectangle_coordinates1 = []
    rectangle_coordinates2 = []

    for halfsec_img1, halfsec_img2, fivesec_img1, fivesec_img2,\
    in zip(camera_halfsec_set1, camera_halfsec_set2, camera_fivesec_set1, camera_fivesec_set2):
        file_5sec_index1, file_halfsec_index1 = five_second_images.index(fivesec_img1), half_second_images.index(halfsec_img1)
        assert file_5sec_index1 == file_halfsec_index1, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'
        file_5sec_index2, file_halfsec_index2 = five_second_images.index(fivesec_img2), half_second_images.index(halfsec_img2)
        assert file_5sec_index2 == file_halfsec_index2, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'

        # searches in all LaunchPlus18Month_PSFImages files for the corresponding index
        all_files_halfsecond_index1, all_files_fivesecond_index1 =  globs.index(halfsec_img1), globs.index(fivesec_img1)
        all_files_halfsecond_index2, all_files_fivesecond_index2 =  globs.index(halfsec_img2), globs.index(fivesec_img2)

        assert all_files_halfsecond_index1 < len(globs) and all_files_fivesecond_index1 < len(globs)
        assert halfsec_img1 not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
        assert halfsec_img2 not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
        assert fivesec_img1 not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'
        assert fivesec_img2 not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'

        # gets the brightest pixel for each image
        brightest_pixel_halfsec1 = brights_halfsec[file_halfsec_index1]
        brightest_pixel_fivesec1 = brights_5s[file_5sec_index1]
        brightest_pixel_halfsec2 = brights_halfsec[file_halfsec_index2]
        brightest_pixel_fivesec2 = brights_5s[file_5sec_index2]

        # first finds the pointing offset coordinates and then matches it to the region
        region_halfsecond1 = assign_region(brightest_pixel_halfsec1)
        region_fivesecond1 = assign_region(brightest_pixel_fivesec1)
        region_halfsecond2 = assign_region(brightest_pixel_halfsec2)
        region_fivesecond2 = assign_region(brightest_pixel_fivesec2)
        assert region_halfsecond1 == region_fivesecond1, 'Regions are not the same!'
        assert region_halfsecond2 == region_fivesecond2, 'Regions are not the same!'

        # for the first set of images
        if region_halfsecond1 in diagonal_paths[diagonal_path] and region_fivesecond1 in diagonal_paths[diagonal_path]:
            replaced_horizontal1, replaced_vertical1 = replace_peaks(file_5sec_index1, cross_section)

            # diagonal coordinates are centered at zero, so we replace them with the coordinates for the horizontal and vertical PSFs
            if cross_section == 'diagonal':
                replaced_horizontal1[0] = replace_peaks(file_5sec_index1,'horizontal_vertical')[0][0]
                replaced_vertical1[0] = replace_peaks(file_5sec_index1,'horizontal_vertical')[1][0]

            normalized_horizontal1 = replaced_horizontal1[1] / radiometric_normalization(file_5sec_index1, cross_section)[0]
            normalized_vertical1 = replaced_vertical1[1] / radiometric_normalization(file_5sec_index1, cross_section)[0]

            ax_horizontal_line1 = ax_horizontal1.plot(normalized_horizontal1, label=region_halfsecond1, linewidth=linewidth, color=region_colors[region_halfsecond1])
            ax_vertical1.plot(normalized_vertical1, label=region_halfsecond1, linewidth=linewidth,color=region_colors[region_fivesecond1])
            colors1.append(ax_horizontal_line1[0].get_color())
            rectangle_coordinates1.append((replaced_horizontal1[0][0]-60,replaced_vertical1[0][-1]-55))

        # duplicate for the second set of images
        if region_halfsecond2 in diagonal_paths[diagonal_path] and region_fivesecond2 in diagonal_paths[diagonal_path]:
            replaced_horizontal2, replaced_vertical2 = replace_peaks(file_5sec_index2, cross_section)

            # diagonal coordinates are centered at zero, so we replace them with the coordinates for the horizontal and vertical PSFs
            if cross_section == 'diagonal':
                replaced_horizontal2[0] = replace_peaks(file_5sec_index2,'horizontal_vertical')[0][0]
                replaced_vertical2[0] = replace_peaks(file_5sec_index2,'horizontal_vertical')[1][0]

            normalized_horizontal2 = replaced_horizontal2[1] / radiometric_normalization(file_5sec_index2, cross_section)[0]
            normalized_vertical2 = replaced_vertical2[1] / radiometric_normalization(file_5sec_index2, cross_section)[0]

            ax_horizontal_line2 = ax_horizontal2.plot(normalized_horizontal2, label=region_halfsecond2, linewidth=linewidth, color=region_colors[region_halfsecond2])
            ax_vertical2.plot(normalized_vertical2, label=region_halfsecond2, linewidth=linewidth,color=region_colors[region_fivesecond2])
            colors2.append(ax_horizontal_line2[0].get_color())
            rectangle_coordinates2.append((replaced_horizontal2[0][0]-60,replaced_vertical2[0][-1]-55))

    # Now for the fun part: adding an inset plot with colored rectangles representing the PSF at each location!
    colored_rectangles_axis1 = inset_axes(ax_horizontal1,3,2)
    colored_rectangles_axis1.imshow(np.zeros([2004,2752]),vmin=0,cmap='gray')
    colored_rectangles_axis1.axis('off')

    colored_rectangles_axis2 = inset_axes(ax_horizontal2,3,2)
    colored_rectangles_axis2.imshow(np.zeros([2004,2752]),vmin=0,cmap='gray')
    colored_rectangles_axis2.axis('off')

    for line_color, left_and_bottom_rectangle in zip(colors1, rectangle_coordinates1):
        colored_rect = patches.Rectangle(left_and_bottom_rectangle, 130,100, edgecolor=line_color,linewidth=1,facecolor='none')
        colored_rectangles_axis1.add_patch(colored_rect)
    for line_color, left_and_bottom_rectangle in zip(colors2, rectangle_coordinates2):
        colored_rect = patches.Rectangle(left_and_bottom_rectangle, 130,100, edgecolor=line_color,linewidth=1,facecolor='none')
        colored_rectangles_axis2.add_patch(colored_rect)


    #ax_horizontal.legend(loc='best'), ax_vertical.legend(loc='best')
    if cross_section == 'horizontal_vertical':
        ax_horizontal1.set_ylabel('Normalized Horizontal PSF',labelpad=14)
        ax_vertical1.set_ylabel('Normalized Vertical PSF', labelpad=14)
    elif cross_section == 'diagonal':
        ax_horizontal1.set_ylabel('Normalized Ascending PSF',labelpad=14)
        ax_vertical1.set_ylabel('Normalized Descending PSF', labelpad=14)
    else:
        return 'Not a correct cross-section'
    #ax_horizontal.set_ylim(top=5000), ax_vertical.set_ylim(top=5000)
    if cross_section == 'horizontal_vertical':
        ax_horizontal2.set_ylabel('Normalized Horizontal PSF',labelpad=14)
        ax_vertical2.set_ylabel('Normalized Vertical PSF', labelpad=14)
    elif cross_section == 'diagonal':
        ax_horizontal2.set_ylabel('Normalized Ascending PSF',labelpad=14)
        ax_vertical2.set_ylabel('Normalized Descending PSF', labelpad=14)
    else:
        return 'Not a correct cross-section'

    directions = {'left_up': 'Northeast', 'left_down': 'Southeast', 'central': 'Central'}
    fig1.suptitle('Camera: {0}   Gain: {1} \n PSF Direction: {2}\nImage Set 1'.format(camera.upper(), gain, directions[diagonal_path]))
    fig2.suptitle('Camera: {0}   Gain: {1}  \n PSF Direction: {2}\nImage Set 2'.format(camera.upper(), gain, directions[diagonal_path]))
    fig1.tight_layout(), fig2.tight_layout()
    #fig.subplots_adjust(left=.06, top=.9,wspace=.22)
    fig1.subplots_adjust(left=.06,right=.98,top=.91,hspace=0)
    fig2.subplots_adjust(left=.06,right=.98,top=.91,hspace=0)

    # makes it only one subplot instead of two
    if diagonal_path == 'left_up':
        ax_vertical1.axis('off'), ax_vertical2.axis('off')
        ax_horizontal1.change_geometry(1,1,1), ax_horizontal2.change_geometry(1,1,1)
    elif diagonal_path == 'left_down':
        ax_horizontal1.axis('off'), ax_horizontal2.axis('off')
        ax_vertical1.change_geometry(1,1,1), ax_vertical2.change_geometry(1,1,1)

    return fig1, fig2

def diagonal_PSF_to_PDFs(output_name, cross_section='horizontal_vertical',camera='NFT'):
    plt.ioff()

    with PdfPages(output_name) as pdf:
        for gain_value in [8,10]:
            for path in ['left_down', 'left_up', 'central']:
                fig1, fig2 = PSF_regions(diagonal_path=path,cross_section=cross_section,camera=camera,gain=gain_value)
                pdf.savefig(fig1)
                pdf.savefig(fig2)
                plt.close(fig1)
                plt.close(fig2)

    plt.ion()

def radiometric_normalization(filenumber, cross_section, half_window_range=5):
    # horizontal_vertical: profile1 is horizontal and profile2 is vertical
    # diagonal_PSFs: profile1 is ascending, profile2 is descending
    profile1, profile2 = replace_peaks(filenumber, cross_section)

    brightest_row, brightest_column = brights_5s[filenumber]

    # for even numbers
    if half_window_range % 2 == 0:
        row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range)
        col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range)
        xfront, xback = col_pixelrange[0], col_pixelrange[1]
        yback, yfront = row_pixelrange[1], row_pixelrange[0]
    # for odd numbers
    elif half_window_range % 2 == 1:
        row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range + 1)
        col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range + 1)
        xfront, xback = col_pixelrange[0], col_pixelrange[1]-1
        yback, yfront = row_pixelrange[1]-1, row_pixelrange[0]

    img_data1 = fits.getdata(five_second_images[filenumber]) / 5
    img_data2 = fits.getdata(five_second_images[filenumber]) / 5

    if cross_section == 'diagonal':
        up_indices, down_indices = diagonal_PSFs(five_second_images[filenumber])[2]
        img_data1[up_indices] = profile1[1]
        profile1_total = np.sum(img_data1[yfront:yback,xfront:xback])

        img_data2[down_indices] = profile2[1]
        profile2_total = np.sum(img_data2[yfront:yback,xfront:xback])

        assert profile1_total < 15000 and profile2_total < 15000, 'Total DN/s too high'
    else:
        return 'Not a valid cross section.'

    return profile1_total, profile2_total

def subwindow_replacement(filenumber, diagonal_path, threshold=760, norm=True):
    assert filenumber not in washed_out_images

    # retrieves the halfsecond and fivesecond data
    halfsecond_data = fits.getdata(half_second_images[filenumber]) / 0.5
    fivesecond_data = fits.getdata(five_second_images[filenumber]) / 5

    # gets the normalization constants from the specific star
    ascending_norm, descending_norm = radiometric_normalization(filenumber, 'diagonal')

    # replace the spots that are saturated in the 5s exposure images
    fivesecond_data[np.where(fits.getdata(five_second_images[filenumber]) > threshold)] = halfsecond_data[np.where(fits.getdata(five_second_images[filenumber]) > threshold)]

    if diagonal_path == 'left_up' and norm == True:
        fivesecond_data /= ascending_norm
    elif diagonal_path == 'left_down' and norm == True:
        fivesecond_data /= descending_norm
    else:
        return 'Did not specify a valid diagonal path'
    return fivesecond_data

'''def PSF_regions(diagonal_path='left_down', cross_section='horizontal_vertical',camera='NFT',gain=8, cmap='tab10',linewidth=1.5, half_window_range=5):
    #Plots the horizontal and vertical PSFs for points that follow the specified diagonal_path
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #DOY100_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY100', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #DOY103_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY103', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #raw_pointingoffsets = np.concatenate((DOY100_pointingoffsets, DOY103_pointingoffsets))
    #pointing_offset_coordinates = coordinates_to_tuples(['({})'.format(string) for string in raw_pointingoffsets if string != 'nan' and string != '---'])

    # specifies the paths taken to plot the diagonal PSFs
    diagonal_paths = {'left_down': ['corner top left', 'top left', 'center', 'bottom right', 'corner bottom right'],
                      'left_up': ['corner bottom left', 'bottom left', 'center', 'top right', 'corner top right'],
                      'central': ['top', 'left', 'bottom', 'right', 'center']}

    assert camera == 'NFT' or camera == 'NCM', 'Invalid camera!'
    assert gain == 8 or gain==10, 'Invalid gain value. We only accept 8 or 10, thank you very much'

    # loads in the images based on gain and camera values
    camera_halfsec = [img for img in half_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(half_second_images)[washed_out_images]]

    camera_fivesec = [img for img in five_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(five_second_images)[washed_out_images]]

    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_halfsec]) == camera).all()
    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_fivesec]) == camera).all()


    # for some reason there's two sets of images!!
    fig, (ax_horizontal, ax_vertical) = plt.subplots(2,1,figsize=(16,9),sharex=True,sharey=True)

    # specifies which color the box around a region shoudl be
    region_colors = {'center': 'tab:red', 'top left': 'orange', 'bottom left': 'orange',
                     'corner top right': 'tab:green', 'corner top left': 'tab:green',
                     'top right': 'tab:blue', 'bottom right': 'tab:blue',
                     'top': 'tab:green', 'bottom': 'tab:purple', 'left': 'orange', 'right':'tab:blue',
                     'corner bottom left': 'tab:purple', 'corner bottom right':'tab:purple'}

    # gets the colors for use in the legend
    colors = []
    rectangle_coordinates = []

    for halfsec_img, fivesec_img in zip(camera_halfsec, camera_fivesec):
        file_5sec_index, file_halfsec_index = five_second_images.index(fivesec_img), half_second_images.index(halfsec_img)
        assert file_5sec_index == file_halfsec_index, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'

        # searches in all LaunchPlus18Month_PSFImages files for the corresponding index
        all_files_halfsecond_index, all_files_fivesecond_index =  globs.index(halfsec_img), globs.index(fivesec_img)


        assert all_files_halfsecond_index < len(globs) and all_files_fivesecond_index < len(globs)
        assert halfsec_img not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
        assert fivesec_img not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'

        # gets the brightest pixel for each image
        brightest_pixel_halfsec = brights_halfsec[file_halfsec_index]
        brightest_pixel_fivesec = brights_5s[file_5sec_index]

        # first finds the pointing offset coordinates and then matches it to the region
        region_halfsecond = assign_region(brightest_pixel_halfsec)
        region_fivesecond = assign_region(brightest_pixel_fivesec)
        assert region_halfsecond == region_fivesecond, 'Regions are not the same!'


        if region_halfsecond in diagonal_paths[diagonal_path] and region_fivesecond in diagonal_paths[diagonal_path]:
            replaced_horizontal, replaced_vertical = replace_peaks(file_5sec_index, cross_section)

            # diagonal coordinates are centered at zero, so we replace them with the coordinates for the horizontal and vertical PSFs
            if cross_section == 'diagonal':
                replaced_horizontal[0] = replace_peaks(file_5sec_index,'horizontal_vertical')[0][0]
                replaced_vertical[0] = replace_peaks(file_5sec_index,'horizontal_vertical')[1][0]

            normalized_horizontal = replaced_horizontal[1] / radiometric_normalization(file_5sec_index, cross_section)[0]
            normalized_vertical = replaced_vertical[1] / radiometric_normalization(file_5sec_index, cross_section)[1]

            ax_horizontal_line = ax_horizontal.plot(normalized_horizontal, label=region_halfsecond, linewidth=linewidth, color=region_colors[region_halfsecond])
            ax_vertical.plot(normalized_vertical, label=region_halfsecond, linewidth=linewidth,color=region_colors[region_fivesecond])
            colors.append(ax_horizontal_line[0].get_color())
            rectangle_coordinates.append((replaced_horizontal[0][0]-60,replaced_vertical[0][-1]-55))

    # Now for the fun part: adding an inset plot with colored rectangles representing the PSF at each location!
    colored_rectangles_axis = inset_axes(ax_horizontal,3,2)
    colored_rectangles_axis.imshow(np.zeros([2004,2752]),vmin=0,cmap='gray')
    colored_rectangles_axis.axis('off')

    for line_color, left_and_bottom_rectangle in zip(colors, rectangle_coordinates):
        colored_rect = patches.Rectangle(left_and_bottom_rectangle, 130,100, edgecolor=line_color,linewidth=1,facecolor='none')
        colored_rectangles_axis.add_patch(colored_rect)


    #ax_horizontal.legend(loc='best'), ax_vertical.legend(loc='best')
    if cross_section == 'horizontal_vertical':
        ax_horizontal.set_ylabel('Normalized Horizontal PSF',labelpad=14)
        ax_vertical.set_ylabel('Normalized Vertical PSF', labelpad=14)
    elif cross_section == 'diagonal':
        ax_horizontal.set_ylabel('Normalized Ascending PSF',labelpad=14)
        ax_vertical.set_ylabel('Normalized Descending PSF', labelpad=14)
    else:
        return 'Not a correct cross-section'

    directions = {'left_up': 'Northeast', 'left_down': 'Southeast', 'central': 'Central'}
    fig.suptitle('Camera: {0}   Gain: {1} \n PSF Direction: {2}'.format(camera.upper(), gain, directions[diagonal_path]))
    fig.tight_layout()
    #fig.subplots_adjust(left=.06, top=.9,wspace=.22)
    fig.subplots_adjust(left=.06,right=.98,top=.91,hspace=0)

    return fig, (ax_horizontal, ax_vertical)'''

directions = {'left_up': 'Northeast', 'left_down': 'Southeast', 'central': 'Central'}

def PSF_regions(diagonal_path='left_down', cross_section='horizontal_vertical',camera='NFT',gain=8, cmap='tab10',linewidth=1.5, half_window_range=5):
    '''Plots the horizontal and vertical PSFs for points that follow the specified diagonal_path
    ONLY WORKS FOR DIAGONAL STUFF THE OTHER STUFF IS HIDDEN'''
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #DOY100_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY100', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #DOY103_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY103', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #raw_pointingoffsets = np.concatenate((DOY100_pointingoffsets, DOY103_pointingoffsets))
    #pointing_offset_coordinates = coordinates_to_tuples(['({})'.format(string) for string in raw_pointingoffsets if string != 'nan' and string != '---'])

    # specifies the paths taken to plot the diagonal PSFs
    diagonal_paths = {'left_down': ['corner top left', 'top left', 'center', 'bottom right', 'corner bottom right'],
                      'left_up': ['corner bottom left', 'bottom left', 'center', 'top right', 'corner top right'],
                      'central': ['top', 'left', 'bottom', 'right', 'center']}

    assert camera == 'NFT' or camera == 'NCM', 'Invalid camera!'
    assert gain == 8 or gain==10, 'Invalid gain value. We only accept 8 or 10, thank you very much'

    # loads in the images based on gain and camera values
    camera_halfsec = [img for img in half_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(half_second_images)[washed_out_images]]

    camera_fivesec = [img for img in five_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(five_second_images)[washed_out_images]]

    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_halfsec]) == camera).all()
    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_fivesec]) == camera).all()


    # for some reason there's two sets of images!!
    fig, ax = plt.subplots(1,1,figsize=(16,9))

    # specifies which color the box around a region shoudl be
    region_colors = {'center': 'tab:red', 'top left': 'orange', 'bottom left': 'orange',
                     'corner top right': 'tab:green', 'corner top left': 'tab:green',
                     'top right': 'tab:blue', 'bottom right': 'tab:blue',
                     'top': 'tab:green', 'bottom': 'tab:purple', 'left': 'orange', 'right':'tab:blue',
                     'corner bottom left': 'tab:purple', 'corner bottom right':'tab:purple'}

    # gets the colors for use in the legend
    colors = []
    rectangle_coordinates = []
    indices_fivesec, indices_halfsec =[], []
    for halfsec_img, fivesec_img in zip(camera_halfsec, camera_fivesec):
        file_5sec_index, file_halfsec_index = five_second_images.index(fivesec_img), half_second_images.index(halfsec_img)
        assert file_5sec_index == file_halfsec_index, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'

        # searches in all LaunchPlus18Month_PSFImages files for the corresponding index
        all_files_halfsecond_index, all_files_fivesecond_index =  globs.index(halfsec_img), globs.index(fivesec_img)


        assert all_files_halfsecond_index < len(globs) and all_files_fivesecond_index < len(globs)
        assert halfsec_img not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
        assert fivesec_img not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'

        # gets the brightest pixel for each image
        brightest_pixel_halfsec = brights_halfsec[file_halfsec_index]
        brightest_pixel_fivesec = brights_5s[file_5sec_index]

        # first finds the pointing offset coordinates and then matches it to the region
        region_halfsecond = assign_region(brightest_pixel_halfsec)
        region_fivesecond = assign_region(brightest_pixel_fivesec)
        assert region_halfsecond == region_fivesecond, 'Regions are not the same!'


        if region_halfsecond in diagonal_paths[diagonal_path] and region_fivesecond in diagonal_paths[diagonal_path]:
            replaced_horizontal, replaced_vertical = replace_peaks(file_5sec_index, cross_section, threshold=770)

            # diagonal coordinates are centered at zero, so we replace them with the coordinates for the horizontal and vertical PSFs
            if cross_section == 'diagonal':
                replaced_horizontal[0] = replace_peaks(file_5sec_index,'horizontal_vertical')[0][0]
                replaced_vertical[0] = replace_peaks(file_5sec_index,'horizontal_vertical')[1][0]
                indices_fivesec.append(file_5sec_index), indices_halfsec.append(file_halfsec_index)
            normalized_horizontal = replaced_horizontal[1] / radiometric_normalization(file_5sec_index, cross_section)[0]
            normalized_vertical = replaced_vertical[1] / radiometric_normalization(file_5sec_index, cross_section)[1]

            if diagonal_path == 'left_up':
                ax_line = ax.plot(np.arange(-5,6), normalized_horizontal, label=region_halfsecond, linewidth=linewidth, color=region_colors[region_halfsecond])
                ax.set_ylabel('Normalized Ascending PSF',labelpad=14)
            elif diagonal_path == 'left_down':
                ax_line = ax.plot(np.arange(-5,6), normalized_vertical, label=region_halfsecond, linewidth=linewidth,color=region_colors[region_fivesecond])
                ax.set_ylabel('Normalized Descending PSF', labelpad=14)
            colors.append(ax_line[0].get_color())
            rectangle_coordinates.append((replaced_horizontal[0][0]-60,replaced_vertical[0][-1]-55))

    # Now for the fun part: adding an inset plot with colored rectangles representing the PSF at each location!
    colored_rectangles_axis = inset_axes(ax,3,2)
    colored_rectangles_axis.imshow(np.zeros([2004,2752]),vmin=0,cmap='gray')
    colored_rectangles_axis.axis('off')

    for line_color, left_and_bottom_rectangle in zip(colors, rectangle_coordinates):
        colored_rect = patches.Rectangle(left_and_bottom_rectangle, 130,100, edgecolor=line_color,linewidth=1,facecolor='none')
        colored_rectangles_axis.add_patch(colored_rect)

    #ax_horizontal.legend(loc='best'), ax_vertical.legend(loc='best')
    ax.set_xlabel('Distance from Center')
    ax.set_ylim(top=0.5)
    '''if cross_section == 'horizontal_vertical':
        ax.set_ylabel('Normalized Horizontal PSF',labelpad=14)
        ax.set_ylabel('Normalized Vertical PSF', labelpad=14)
    elif cross_section == 'diagonal':
        ax.set_ylabel('Normalized Ascending PSF',labelpad=14)
        ax.set_ylabel('Normalized Descending PSF', labelpad=14)
    else:
        return 'Not a correct cross-section'''

    directions = {'left_up': 'Northeast', 'left_down': 'Southeast', 'central': 'Central'}
    fig.suptitle('Camera: {0}   Gain: {1} \n PSF Direction: {2}'.format(camera.upper(), gain, directions[diagonal_path]))
    fig.tight_layout()
    #fig.subplots_adjust(left=.06, top=.9,wspace=.22)
    fig.subplots_adjust(left=.06,right=.98,top=.93,hspace=0)

    '''# makes it only one subplot instead of two
    if diagonal_path == 'left_up':
        ax_horizontal.change_geometry(1,1,1)
        ax_vertical.axis('off')
    elif diagonal_path == 'left_down':
        ax_vertical.change_geometry(1,1,1)
        ax_horizontal.axis('off')'''

    return fig

def diagonal_PSF_to_PDFs(output_name, cross_section='horizontal_vertical',camera='NFT'):
    plt.ioff()

    with PdfPages(output_name) as pdf:
        for gain_value in [8,10]:
            for path in ['left_down', 'left_up']:
                fig = PSF_regions(diagonal_path=path,cross_section=cross_section,camera=camera,gain=gain_value)
                pdf.savefig(fig)
                plt.close(fig)
    plt.ion()

#diagonal_PSF_to_PDFs('18 Month PSFs/NFT Diagonal PSF Regions radiometric normalization.pdf', 'diagonal','NFT')
#diagonal_PSF_to_PDFs('18 Month PSFs/NCM Diagonal PSF Regions with radiometric normalization.pdf', 'diagonal','NCM')
#diagonal_PSF_to_PDFs('Diagonal PSF Regions DOY 103 NFT.pdf', 'diagonal', 103,'nft')

def surface_plot(file_index, plot=False, half_window_range=5, style='fivethirtyeight', colormap='coolwarm', title_fontsize=12, axis_fontsize=10, labelpad=10, vmin_vmax=[100,2500]):
    from mpl_toolkits.mplot3d import Axes3D

    assert file_index <= 77
    assert file_index not in washed_out_images
    plt.style.use(style)
    data_halfsec, data_fivesec = fits.getdata(half_second_images[file_index]), fits.getdata(five_second_images[file_index])

    [xfront_halfsec, xback_halfsec, yfront_halfsec, yback_halfsec] = horizontal_vertical_PSF(half_second_images[file_index],half_window_range=half_window_range,exposure=0.5)[3]
    [xfront_fivesec, xback_fivesec, yfront_fivesec, yback_fivesec] = horizontal_vertical_PSF(five_second_images[file_index],half_window_range=half_window_range,exposure=5)[3]

    x_halfsec, y_halfsec = np.arange(xfront_halfsec, xback_halfsec), np.arange(yfront_halfsec, yback_halfsec)
    x_fivesec, y_fivesec = np.arange(xfront_fivesec, xback_fivesec), np.arange(yfront_fivesec, yback_fivesec)
    # unsure why meshgrid is needed, but it seems like it is
    x_halfsec, y_halfsec = np.meshgrid(x_halfsec, y_halfsec)
    x_fivesec, y_fivesec = np.meshgrid(x_fivesec, y_fivesec)

    focused_halfsec = data_halfsec[yfront_halfsec:yback_halfsec, xfront_halfsec:xback_halfsec]
    focused_fivesec = data_fivesec[yfront_fivesec:yback_fivesec, xfront_fivesec:xback_fivesec]

    if plot:
        fig = plt.figure(figsize=(11,8.5))
        #ax_image = fig.add_subplot(211)
        surface_halfsec = fig.add_subplot(121,projection='3d')
        surface_fivesec = fig.add_subplot(122,projection='3d')

        '''# all this for just the one image at the top
        import matplotlib.patches as patches
        full_img = fits.getdata(five_second_images[file_index])
        brightest_row, brightest_column = brights_5s[file_index]
        rectangle = patches.Rectangle((brightest_column-65,brightest_row-50), 130,100, edgecolor='r',linewidth=1,facecolor='none')
        # plot the full image and the rectangle around the location
        ax_image.imshow(full_img,vmin=vmin_vmax[0]-80, vmax=vmin_vmax[1]-2200, cmap='gray', aspect='equal')
        ax_image.add_patch(rectangle)
        ax_image.set_title('0.5s image: {1} \n 5s image: {0}'.format(five_second_images[file_index][27:],half_second_images[file_index][27:]),fontsize=title_fontsize, pad=15)
        ax_image.set_xlabel('Pixel (x)',fontsize=axis_fontsize), ax_image.set_ylabel('Pixel (y)',fontsize=axis_fontsize)
        ax_image.grid(False)'''

        surf1=surface_halfsec.plot_surface(x_halfsec, y_halfsec, focused_halfsec, cmap=colormap)
        surf2=surface_fivesec.plot_surface(x_fivesec, y_fivesec, focused_fivesec, cmap=colormap)

        surface_halfsec.view_init(elev=16, azim=40.5), surface_fivesec.view_init(elev=16, azim=40.5)

        surface_halfsec.set_title('0.5s: {}'.format(half_second_images[file_index][27:]), fontsize=title_fontsize)
        surface_halfsec.set_xlabel('Pixel (x)', fontsize=axis_fontsize, labelpad=labelpad), surface_halfsec.set_ylabel('Pixel (y)', fontsize=axis_fontsize, labelpad=labelpad)

        surface_fivesec.set_title('5s: {}'.format(five_second_images[file_index][27:]), fontsize=title_fontsize)
        surface_fivesec.set_xlabel('Pixel (x)', fontsize=axis_fontsize, labelpad=labelpad), surface_fivesec.set_ylabel('Pixel (y)', fontsize=axis_fontsize, labelpad=labelpad)
        surface_halfsec.tick_params(labelsize=axis_fontsize,pad=labelpad-5), surface_fivesec.tick_params(labelsize=axis_fontsize,pad=labelpad-5)
        surface_halfsec.set_zlim(top=3950), surface_fivesec.set_zlim(top=3950)

        # creates a separate axis for the colorbar and adjusts it accordingly within the figure
        ax_colorbar = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        ax_colorbar.tick_params(labelsize=axis_fontsize)
        fig.colorbar(surf2, cax=ax_colorbar)

        fig.tight_layout()
        fig.subplots_adjust(right=0.88, wspace=.02)

        return fig, (surface_halfsec, surface_fivesec), (x_halfsec, y_halfsec, focused_halfsec), (x_fivesec, y_fivesec, focused_fivesec)

    else:
        return (x_halfsec, y_halfsec, focused_halfsec), (x_fivesec, y_fivesec, focused_fivesec)

def PDF_surface_and_contour_plots(output_name, plotstyle='surface'):
    plt.ioff()
    image_set = [img for img in np.arange(len(five_second_images)) if img not in washed_out_images]

    with PdfPages(output_name) as pdf:
        for filenumber in image_set:
            if plotstyle == 'surface':
                active_figure = surface_plot(filenumber)[0]
            elif plotstyle == 'contour':
                active_figure = contour_plotting(filenumber)[0]
            else:
                return 'No matching plot style'
            pdf.savefig(active_figure)
            plt.close()

    plt.ion()

def contour_plotting(file_index, plot=True, style = 'fivethirtyeight',colormap='viridis', title_fontsize=12, axis_fontsize=10, labelpad=10, vmin_vmax=[100,2500]):
    # pull the data from the surface_plot function, since it is the same parameters needed for contour plotting
    (x_halfsec, y_halfsec, focused_halfsec), (x_fivesec, y_fivesec, focused_fivesec) = surface_plot(file_index)

    fig, (ax_contourhalf, ax_contourfive) = plt.subplots(1,2, figsize=(12,8))
    plt.style.use(style)

    min_pixel_value = np.min(focused_fivesec)
    max_pixel_value = np.max(focused_fivesec)

    contourhalf = ax_contourhalf.contour(x_halfsec, y_halfsec, focused_halfsec, cmap=colormap,vmin=0, vmax=4000)
    contourfive = ax_contourfive.contour(x_fivesec, y_fivesec, focused_fivesec, cmap=colormap,vmin=0, vmax=4000)

    ax_contourhalf.set_title('0.5s: {}'.format(half_second_images[file_index][27:]), fontsize=title_fontsize, pad=labelpad)
    ax_contourhalf.set_xlabel('Pixel (x)', fontsize=axis_fontsize), ax_contourhalf.set_ylabel('Pixel (y)', fontsize=axis_fontsize)
    ax_contourhalf.tick_params(labelsize=axis_fontsize)

    ax_contourfive.set_title('5s: {}'.format(five_second_images[file_index][27:]), fontsize=title_fontsize, pad=labelpad)
    ax_contourfive.set_xlabel('Pixel (x)', fontsize=axis_fontsize), ax_contourfive.set_ylabel('Pixel (y)', fontsize=axis_fontsize)
    ax_contourfive.tick_params(labelsize=axis_fontsize)

    ax_contourhalf.set_ylim(ax_contourhalf.get_ylim()[::-1])
    ax_contourfive.set_ylim(ax_contourfive.get_ylim()[::-1])

    ax_contourhalf.tick_params(labelsize=axis_fontsize,pad=labelpad)
    ax_contourfive.tick_params(labelsize=axis_fontsize,pad=labelpad)

    # creates a separate axis for the colorbar and adjusts it accordingly within the figure
    ax_colorbar = fig.add_axes([0.90, 0.15, 0.03, 0.7])
    ax_colorbar.tick_params(labelsize=axis_fontsize)
    fig.colorbar(contourfive, cax=ax_colorbar)

    fig.tight_layout()
    fig.subplots_adjust(right=0.88, wspace=.24)

    '''Parameters to think about: semimajor axis, eccentricity'''
    return fig, (ax_contourhalf, ax_contourfive), [contourhalf, contourfive]

#PDF_surface_and_contour_plots('regular contour plots.pdf', 'contour')
region_colors = {'center': 'tab:red', 'top left': 'orange', 'bottom left': 'orange',
                 'corner top right': 'tab:green', 'corner top left': 'tab:green',
                 'top right': 'tab:blue', 'bottom right': 'tab:blue',
                 'top': 'tab:green', 'bottom': 'tab:purple', 'left': 'orange', 'right':'tab:blue',
                 'corner bottom left': 'tab:purple', 'corner bottom right':'tab:purple'}

# explicit coordinates for the regions and maps them to region name
rectangle_centers = {(1043,1466): 'center', (171,321):'corner top left', (171,2607):'corner top right',
                     (1863,313): 'corner bottom left', (1863,2589):'corner bottom right',
                     (573, 2120):'top right', (1043, 2120): 'right', (1517, 2120):'bottom right',
                     (1517, 800):'bottom left', (1043, 800):'left', (573, 800):'top left',
                     (1517, 1466):'bottom', (573, 1466):'top'}
# flips the dictionary so all key-value pairs are reversed
flipped_rectangle_centers = {v: k for k, v in rectangle_centers.items()}

def rectangle_plots(x_location=[70,115], y_location=[163,38], origin_location=[15,45], box_dimensions=[180,180], vmin_vmax=[0,30],output_name=None):
    '''Creates a plot of with a bunch of rectangular regions that represent PSF regions.
    Also indicates the coordinate system that will be used for the presentation.
    For standard rectangles, box_dimensions=[130,100]'''

    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax.grid(False)

    # create a rectangle 2752 x 2004 image rectangle representing a sample image
    sample_image_file = five_second_images[31]
    ax.imshow(fits.getdata(sample_image_file),vmin=vmin_vmax[0],vmax=vmin_vmax[1],cmap='gray')

    for coordinate in rectangle_centers:
        brightest_row, brightest_column = coordinate[0], coordinate[1]
        rect_color = region_colors[rectangle_centers[coordinate]]
        rectangle = patches.Rectangle((brightest_column-(box_dimensions[0]/2),brightest_row-(box_dimensions[1]/2)), box_dimensions[0],box_dimensions[1], edgecolor=rect_color,linewidth=0.7,facecolor='none')
        _= ax.add_patch(rectangle)
    ax.arrow(134,50,100,0, color='red', head_width=15)
    ax.arrow(134,50,0,100, color='red', head_width=15)
    ax.set(xticks=[],yticks=[])
    ax.annotate('+X', xy=x_location, xycoords='data', color='red', fontname='Myriad Pro')
    ax.annotate('+Y', xy=y_location, xycoords='data', color='red', fontname='Myriad Pro')
    ax.annotate('(0,0)', xy=origin_location, xycoords='data', color='white', fontname='Myriad Pro')
    fig.tight_layout()

    if output_name is not None:
        with PdfPages(output_name) as pdf:
            pdf.savefig(fig, dpi=1000)
    return fig, ax

def FOV_hybrids(camera='NFT', gain=8, box_dimensions=[180,180], vmin_vmax=[0,30]):
    fig, ax = rectangle_plots(box_dimensions=box_dimensions, vmin_vmax=vmin_vmax)
    box_half = box_dimensions[0] / 2

    for directionality in ['left_up', 'left_down', 'central']:
        [hybrids, filtered_regions_fivesec] = visualize_PSFs(diagonal_path=directionality,camera=camera,gain=gain,plot=False)[1]
        # plots the PSF image onto its respective region in the big rectangular image
        for i in np.arange(len(hybrids)):
            # gets the location of where to put the PSF image, in data coordinates of the big rectangle (row,column)
            PSF_location = flipped_rectangle_centers[filtered_regions_fivesec[i]]

            # sets that location with the extent (left, right, bottom, top)
            extent = [PSF_location[1]-box_half, PSF_location[1]+box_half, PSF_location[0]+box_half, PSF_location[0]-box_half]
            ax.imshow(hybrids[i],extent=extent, cmap='gray')
    # needs to be set as the last line to display the full image, otherwise will only display the last imshow instance
    ax.set(xlim=[0,2752],ylim=(2004,0))
    ax.set_title('Launch+18 Months',fontsize=12), ax.set_title('Camera: {}'.format(camera), loc='left',fontsize=12), ax.set_title('Gain: {}'.format(gain), loc='right',fontsize=12)
    fig.tight_layout()
    return fig

def comparison_darkcorrection(filename, half_window_range=5, save=False):
    '''Takes in a filename (preferably in the corrected images) and presents it side by side
    with the uncorrected image to show that the dark correction worked.'''

    uncorrected_img = fits.getdata('LaunchPlus18Month_PSFImages/{}'.format(filename[34:]))
    corrected_img = fits.getdata(filename)
    exposure = np.round(fits.getheader(filename)['EXPTIME'],2)

    assert fits.getheader(filename)['DATE_OBS'] == fits.getheader('LaunchPlus18Month_PSFImages/{}'.format(filename[34:]))['DATE_OBS']
    assert exposure == 0.5 or exposure == 5, 'Exposure is not 0.5s or 5s.'

    if exposure == 0.5:
        file_index = half_second_images.index(filename)
        assert half_second_images[file_index][34:53].replace('S','')==fits.getheader(filename)['DATE_OBS'].replace('-','').replace(':','').replace('.',''), \
        'File is not in five_second_exposures or doesn\'t match'
        brightest_row, brightest_column = brights_halfsec[file_index]
        assert np.round(fits.getheader(half_second_images[file_index])['EXPTIME'],2) == exposure
        vmin_vmax = [0,1000]
    elif exposure == 5:
        file_index = five_second_images.index(filename)
        assert five_second_images[file_index][34:53].replace('S','')==fits.getheader(filename)['DATE_OBS'].replace('-','').replace(':','').replace('.',''), \
        'File is not in five_second_exposures or doesn\'t match'
        brightest_row, brightest_column = brights_5s[file_index]
        assert np.round(fits.getheader(five_second_images[file_index])['EXPTIME'],2) == exposure
        vmin_vmax = [0,3500]
    else:
        return 'No match for that exposure'

    # for even numbers
    if half_window_range % 2 == 0:
        row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range)
        col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range)
        xfront, xback = col_pixelrange[0], col_pixelrange[1]
        yback, yfront = row_pixelrange[1], row_pixelrange[0]
    # for odd numbers
    elif half_window_range % 2 == 1:
        row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range + 1)
        col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range + 1)
        xfront, xback = col_pixelrange[0], col_pixelrange[1]-1
        yback, yfront = row_pixelrange[1]-1, row_pixelrange[0]

    fig, (ax_uncorrected, ax_corrected) = plt.subplots(1,2,figsize=(10,8))
    ax_uncorrected.imshow(uncorrected_img, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='gray')
    ax_uncorrected.set(xlim=[xfront, xback], ylim=[yback, yfront], title='{}s Exposure Raw Image'.format(exposure))
    ax_uncorrected.axis('off')
    ax_corrected.imshow(corrected_img, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap='gray')
    ax_corrected.set(xlim=[xfront, xback], ylim=[yback, yfront], title='Dark Corrected')
    ax_corrected.axis('off')
    fig.tight_layout()

    if save:
        fig.savefig(filename[34:]+ ' Raw and Dark Corrected.png')

def PSF_timecomparison(diagonal_path='left_down', cross_section='diagonal',camera='NFT', DOY=100,gain=8, cmap='tab10',linewidth=1.5, half_window_range=5):
    '''Plots the horizontal and vertical PSFs for points that follow the specified diagonal_path
    ONLY WORKS FOR DIAGONAL STUFF THE OTHER STUFF IS HIDDEN'''
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    labelpad = 10
    #DOY100_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY100', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #DOY103_pointingoffsets = pd.read_excel('TAGCAMS imaging order actual.xlsx', 'DOY103', skiprows=6)['POINTING OFFSET (°)'].values.astype(str)
    #raw_pointingoffsets = np.concatenate((DOY100_pointingoffsets, DOY103_pointingoffsets))
    #pointing_offset_coordinates = coordinates_to_tuples(['({})'.format(string) for string in raw_pointingoffsets if string != 'nan' and string != '---'])

    # specifies the paths taken to plot the diagonal PSFs
    diagonal_paths = {'left_down': ['corner top left', 'top left', 'center', 'bottom right', 'corner bottom right'],
                      'left_up': ['corner bottom left', 'bottom left', 'center', 'top right', 'corner top right'],
                      'central': ['top', 'left', 'bottom', 'right', 'center']}

    assert camera == 'NFT' or camera == 'NCM', 'Invalid camera!'
    assert gain == 8 or gain==10, 'Invalid gain value. We only accept 8 or 10, thank you very much'

    # loads in the images based on gain and camera values
    camera_halfsec = [img for img in half_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(half_second_images)[washed_out_images]]

    camera_fivesec = [img for img in five_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera\
    and img not in np.array(five_second_images)[washed_out_images]]

    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_halfsec]) == camera).all()
    assert (np.array([fits.getheader(file)['INSTRUME'] for file in camera_fivesec]) == camera).all()


    # for some reason there's two sets of images!!
    fig, ax = plt.subplots(1,1,figsize=(16,9))
    ax.change_geometry(1,2,2)

    # specifies which color the box around a region shoudl be
    region_colors = {'center': 'tab:red', 'top left': 'orange', 'bottom left': 'orange',
                     'corner top right': 'tab:green', 'corner top left': 'tab:green',
                     'top right': 'tab:blue', 'bottom right': 'tab:blue',
                     'top': 'tab:green', 'bottom': 'tab:purple', 'left': 'orange', 'right':'tab:blue',
                     'corner bottom left': 'tab:purple', 'corner bottom right':'tab:purple'}

    # gets the colors for use in the legend
    colors = []
    rectangle_coordinates = []
    indices_fivesec, indices_halfsec =[], []
    for halfsec_img, fivesec_img in zip(camera_halfsec, camera_fivesec):
        file_5sec_index, file_halfsec_index = five_second_images.index(fivesec_img), half_second_images.index(halfsec_img)
        assert file_5sec_index == file_halfsec_index, 'Index of 5s exposure image and 0.5s exposure image don\'t match up'

        # searches in all LaunchPlus18Month_PSFImages files for the corresponding index
        all_files_halfsecond_index, all_files_fivesecond_index =  globs.index(halfsec_img), globs.index(fivesec_img)


        assert all_files_halfsecond_index < len(globs) and all_files_fivesecond_index < len(globs)
        assert halfsec_img not in np.array(half_second_images)[washed_out_images], 'Index is in the 0.5s washed_out_images'
        assert fivesec_img not in np.array(five_second_images)[washed_out_images], 'Index is in the 5s washed_out_images'

        # gets the brightest pixel for each image
        brightest_pixel_halfsec = brights_halfsec[file_halfsec_index]
        brightest_pixel_fivesec = brights_5s[file_5sec_index]

        # first finds the pointing offset coordinates and then matches it to the region
        region_halfsecond = assign_region(brightest_pixel_halfsec)
        region_fivesecond = assign_region(brightest_pixel_fivesec)
        assert region_halfsecond == region_fivesecond, 'Regions are not the same!'


        if region_halfsecond in diagonal_paths[diagonal_path] and region_fivesecond in diagonal_paths[diagonal_path]:
            replaced_horizontal, replaced_vertical = replace_peaks(file_5sec_index, cross_section, threshold=770)

            # diagonal coordinates are centered at zero, so we replace them with the coordinates for the horizontal and vertical PSFs
            if cross_section == 'diagonal':
                replaced_horizontal[0] = replace_peaks(file_5sec_index,'horizontal_vertical')[0][0]
                replaced_vertical[0] = replace_peaks(file_5sec_index,'horizontal_vertical')[1][0]
                indices_fivesec.append(file_5sec_index), indices_halfsec.append(file_halfsec_index)
            normalized_horizontal = replaced_horizontal[1] / radiometric_normalization(file_5sec_index, cross_section)[0]
            normalized_vertical = replaced_vertical[1] / radiometric_normalization(file_5sec_index, cross_section)[1]

            if diagonal_path == 'left_up':
                ax_line = ax.plot(np.arange(-5,6), normalized_horizontal, label=region_halfsecond, linewidth=linewidth, color=region_colors[region_halfsecond])
                ax.set_ylabel('Normalized Ascending PSF (1/s)',labelpad=labelpad)
            elif diagonal_path == 'left_down':
                ax_line = ax.plot(np.arange(-5,6), normalized_vertical, label=region_halfsecond, linewidth=linewidth,color=region_colors[region_fivesecond])
                ax.set_ylabel('Normalized Descending PSF (1/s)', labelpad=labelpad)
            colors.append(ax_line[0].get_color())
            rectangle_coordinates.append((replaced_horizontal[0][0]-60,replaced_vertical[0][-1]-55))

    # Now for the fun part: adding an inset plot with colored rectangles representing the PSF at each location!
    colored_rectangles_axis = plt.axes([0.804,0.75,0.22,0.17])
    colored_rectangles_axis.imshow(np.zeros([2004,2752]),vmin=0,cmap='gray')
    colored_rectangles_axis.axis('off')

    for line_color, left_and_bottom_rectangle in zip(colors, rectangle_coordinates):
        colored_rect = patches.Rectangle(left_and_bottom_rectangle, 130,100, edgecolor=line_color,linewidth=1,facecolor='none')
        colored_rectangles_axis.add_patch(colored_rect)


    #ax_horizontal.legend(loc='best'), ax_vertical.legend(loc='best')
    ax.set_xlabel('Distance from Center')
    ax.set_ylim(top=0.5)
    '''if cross_section == 'horizontal_vertical':
        ax.set_ylabel('Normalized Horizontal PSF',labelpad=14)
        ax.set_ylabel('Normalized Vertical PSF', labelpad=14)
    elif cross_section == 'diagonal':
        ax.set_ylabel('Normalized Ascending PSF',labelpad=14)
        ax.set_ylabel('Normalized Descending PSF', labelpad=14)
    else:
        return 'Not a correct cross-section'''
    #directions[diagonal_path]
    directions = {'left_up': 'Northeast', 'left_down': 'Southeast', 'central': 'Central'}
    ax.set_title('+18 Months')
    fig.tight_layout()
    fig.suptitle('Camera: {0}  Gain: {1}\n PSF Direction: {2}'.format(camera, gain, directions[diagonal_path]))
    #fig.subplots_adjust(left=.06, top=.9,wspace=.22)
    fig.subplots_adjust(left=.06,right=.98,top=.92,hspace=0)

    '''# makes it only one subplot instead of two
    if diagonal_path == 'left_up':
        ax_horizontal.change_geometry(1,1,1)
        ax_vertical.axis('off')
    elif diagonal_path == 'left_down':
        ax_vertical.change_geometry(1,1,1)
        ax_horizontal.axis('off')'''

    # Plot the subwindows for each of the things plotted
    num_lines = len(ax.lines)
    fig_zoomedimages = plt.figure(figsize=[7.38,5.35])

    assert num_lines % 5==0 or num_lines % 4 ==0, 'Unexpected number of lines!'

    previously_marked_regions = []
    zoomed_img_count = 1

    for fivesec_img in camera_fivesec:
        file_5sec_index = five_second_images.index(fivesec_img)
        brightest_pixel_fivesec = brights_5s[file_5sec_index]
        region_fivesecond = assign_region(brightest_pixel_fivesec)

        if region_fivesecond in diagonal_paths[diagonal_path] and region_fivesecond not in previously_marked_regions:
            assert file_5sec_index < len(five_second_images)
            assert file_5sec_index not in washed_out_images

            brightest_row, brightest_column = brightest_pixel_fivesec

            # for even numbers
            if half_window_range % 2 == 0:
                row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range)
                col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range)
                xfront, xback = col_pixelrange[0], col_pixelrange[1]
                yback, yfront = row_pixelrange[1], row_pixelrange[0]
            # for odd numbers
            elif half_window_range % 2 == 1:
                row_pixelrange = (brightest_row - half_window_range, brightest_row + half_window_range + 1)
                col_pixelrange = (brightest_column - half_window_range, brightest_column + half_window_range + 1)
                xfront, xback = col_pixelrange[0], col_pixelrange[1]-1
                yback, yfront = row_pixelrange[1]-1, row_pixelrange[0]

            if num_lines % 5==0:
                ax_zoomedimg = fig_zoomedimages.add_subplot(2,3,zoomed_img_count)
            elif num_lines % 4 ==0:
                ax_zoomedimg= fig_zoomedimages.add_subplot(2,2,zoomed_img_count)
            else:
                return 'Unexpected number of lines'
            ax_zoomedimg.imshow(subwindow_replacement(file_5sec_index, diagonal_path, threshold=760), vmin=0, vmax=0.25, cmap='gray')

            # zooms in on the image and removes the ticks and grid pattern
            ax_zoomedimg.set(xlim=[xfront, xback], ylim=[yback, yfront], xticks=[], yticks=[])
            ax_zoomedimg.grid(False)

            # sets the color of the spine
            spine_color = region_colors[region_fivesecond]
            for border in ['top', 'bottom', 'left', 'right']:
                ax_zoomedimg.spines[border].set_color(spine_color)
                ax_zoomedimg.spines[border].set_linewidth(2)

            fig_zoomedimages.tight_layout()
            fig_zoomedimages.subplots_adjust(wspace=0.11)
            previously_marked_regions.append(region_fivesecond)
            zoomed_img_count += 1
    return fig, fig_zoomedimages

def find_centroid(filename, threshold=780):
    #import time
    #start_time = time.time()
    exposure = np.round(fits.getheader(filename)['EXPTIME'],2)
    if exposure == 0.5:
        row,column = brights_halfsec[half_second_images.index(filename)]
    elif exposure == 5:
        row,column = brights_halfsec[five_second_images.index(filename)]
    else:
        return 'Exposure times do not match 0.5s or 5s.'
    [d,row_pixelrange,col_pixelrange] = horizontal_vertical_PSF(filename, exposure, plot=False)[2]
    [xfront, xback, yfront, yback] = horizontal_vertical_PSF(filename, exposure, plot=False)[3]
    row_pixelrange = np.arange(row_pixelrange[0]+1, row_pixelrange[-1])
    col_pixelrange = np.arange(col_pixelrange[0]+1, col_pixelrange[-1])

    # Loads the data for that file and normalizes by exposure
    d /= exposure
    assert column in col_pixelrange, 'Column not in x-range'
    assert row in row_pixelrange, 'Row not in y-range'

    # reduces the data to just the subwindow
    subwindow = d[yfront:yback, xfront:xback]

    # replaces oversaturated pixels with 0 in the 5s images
    if exposure == 5:
        d[d > threshold] = 0
    DN_rowsum = np.sum(subwindow,1)
    DN_colsum = np.sum(subwindow,0)

    centroid_row = np.sum(row_pixelrange*DN_rowsum) / np.sum(DN_rowsum)
    centroid_col = np.sum(col_pixelrange*DN_colsum) / np.sum(DN_colsum)
    #print ('Runtime: {} seconds'.format(time.time()-start_time))
    return (centroid_row,centroid_col), (row,column), subwindow

def calculate_distance(vector1, vector2):
    # Calculates the Euclidean distance between two vectors via the well-known distance formula
    if type(vector1) == list:
        vector1 = np.array(vector1)
    if type(vector2) == list:
        vector2 = np.array(vector2)
    distance = np.linalg.norm(np.subtract(vector1,vector2), 2, 0)
    return distance

def distances_between_centroids():
    '''Currently finds distance between SATURATED centroids'''
    centroids_half = [find_centroid(file,900)[0] for file in half_second_images]
    centroids_five = [find_centroid(file,900)[0] for file in five_second_images]
    return centroids_half, centroids_five, [calculate_distance(v1,v2) for v1,v2 in zip(centroids_half,centroids_five)]

'''centroids_half, centroids_five, centroid_distances6month = distances_between_centroids()
df=pd.DataFrame(np.array([centroids_half, centroids_five, centroid_distances6month]).T,columns=['0.5s Centroids (saturated)','5s Centroids (saturated)', 'Centroid Distances (saturated)'])
with pd.ExcelWriter('6month no saturated centroids.xlsx') as writer:
    df.to_excel(writer)
    writer.save()'''

def resample_image(filename, end_size=[1000,1000], order=3):
    #import time
    #start_time = time.time()
    '''Resamples the image using bicubic interpolation. At time of writing, this will expand resolution
    from 10x10 subwindows to 100x100 subwindows.'''
    assert filename not in np.array(half_second_images)[washed_out_images] and filename not in np.array(five_second_images)[washed_out_images]

    # load the image data as well as exposure time and subwindow coordinates
    exposure = np.round(fits.getheader(filename)['EXPTIME'],2)
    [data,row_pixelrange,col_pixelrange] = horizontal_vertical_PSF(filename, exposure, plot=False)[2]
    [xfront, xback, yfront, yback] = horizontal_vertical_PSF(filename, exposure, plot=False)[3]
    zoomed_img = data[yfront:yback, xfront:xback]


    # resample the images to obtain higher pixel resolution using one of the interpolation methods below
    # order ranges from 0 to 5: nearest-neighbor, bilinear, bicubic, biquartic, biquintic
    resampled = transform.resize(zoomed_img, end_size, order=order,mode='reflect',anti_aliasing=True)
    #print ('Runtime: {} seconds'.format(time.time()-start_time))
    return zoomed_img, resampled

'''plt.imshow(resample_image(five_second_images[3],'left_up')[0], cmap='gray')
plt.imshow(resample_image(five_second_images[3],'left_up')[1], cmap='gray')
plt.imshow(resample_image(five_second_images[3],'left_up',4)[1], cmap='gray')
plt.imshow(resample_image(five_second_images[3],'left_up',5)[1], cmap='gray')'''

centroids_half = np.asarray(coordinates_to_tuples(pd.read_excel('Coordinates to Brightest Pixel.xlsx')['0.5s Centroids (unsaturated)']))
centroids_five = np.asarray(coordinates_to_tuples(pd.read_excel('Coordinates to Brightest Pixel.xlsx')['5s Centroids (unsaturated)']))
centroid_differences = centroids_five - centroids_half

def find_shifts(filenumber):
    '''Uses a cross-correlation function to do some fancy convolution and phase correlation.
    Outputs shift required to move the 0.5s image to fit over the 5s image.'''
    from skimage.feature import register_translation
    original_halfsecond_img, resampled_halfsecond_img = resample_image(half_second_images[filenumber],end_size=[100,100])
    original_fivesecond_img, resampled_fivesecond_img = resample_image(five_second_images[filenumber],end_size=[100,100])

    shift, error, diffphase = register_translation(original_fivesecond_img,original_halfsecond_img,upsample_factor=100)
    return shift, error, diffphase

def shift_images(resampled_data, shift_rows, shift_columns):
    '''Takes an array of hopefully resampled data and then shifts by the number of rows and columns needed'''
    data_shiftedrows = np.roll(resampled_data, shift_rows, axis=0)
    data_shiftedcolumns = np.roll(data_shiftedrows, shift_columns, axis=1)

    return data_shiftedcolumns

def realignment(filenumber, sampling_size=[1000,1000]):
    '''Upsamples the 0.5s and 5s images, then shifts them by the centroid difference to hopefully align them.
    Centroid of 0.5s image is shift to centroid of 5s image.'''
    assert sampling_size[0] == sampling_size[1]

    # uses bicubic interpolation to resample images. I believe we are going for 1000x1000 images, also normalized
    original_half, resampled_half = resample_image(half_second_images[filenumber], end_size=sampling_size)
    original_half /= 0.5
    resampled_half /= 0.5
    original_five, resampled_five = resample_image(five_second_images[filenumber], end_size=sampling_size)
    original_five /= 5
    resampled_five /= 5
    # obtains the number of rows and columns to shift from the previously defined centroid differences array
    # also scales the shifts to integer values
    shift = find_shifts(filenumber)[0]
    shift_rows = int(np.round(shift[0] * (sampling_size[0]/10)))
    shift_columns = int(np.round(shift[1] * (sampling_size[0]/10)))

    # shift the 0.5s image onto the 5s image
    shifted_halfsecond_image = shift_images(resampled_half, shift_rows, shift_columns)

    # downsample back to the 10x10 subwindow size
    downsampled = transform.resize(shifted_halfsecond_image,[10,10], order=3, mode='reflect', anti_aliasing=True)
    return shifted_halfsecond_image, resampled_half, resampled_five, [original_half, original_five], downsampled

def plot_shifted(filenumber):
    fig,([ax1,ax2,ax3],[ax4,ax5,ax6], [ax7,ax8,ax9]) = plt.subplots(3,3, figsize=(12,8))
    shifted_halfsecond_image, resampled_half, resampled_five, [original_half,original_five], downsampled = realignment(filenumber)

    ax1.imshow(resampled_half,cmap='gray',vmin=0,vmax=2500)
    ax2.imshow(resampled_five,cmap='gray',vmin=0,vmax=2500)
    ax3.imshow(shifted_halfsecond_image,cmap='gray',vmin=0,vmax=2500)
    ax4.imshow(original_half,cmap='gray',vmin=0,vmax=2500)
    ax5.imshow(original_five,cmap='gray',vmin=0,vmax=2500)
    ax6.imshow(downsampled,cmap='gray',vmin=0,vmax=2500)
    ax8.imshow(hybrid_PSF(filenumber),cmap='gray',vmin=0,vmax=2500)
    ax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
    for a in ax:
        a.grid(False)
    ax7.axis('off'), ax9.axis('off')
    fig.tight_layout()

def hybrid_PSF(filenumber, threshold=780):
    import time
    start_time = time.time()
    [original_half,original_five], downsampled_shifted_image = realignment(filenumber)[-2:]
    original_five[original_five > threshold] = downsampled_shifted_image[original_five > threshold]
    print ('Runtime: {} seconds'.format(time.time()-start_time))
    return original_five

def visualize_PSFs(diagonal_path='left_up',camera='NFT',gain=8,plot=True):
    '''Plots the horizontal and vertical PSFs for points that follow the specified diagonal_path'''
    #plt.style.use('bmh')
    # specifies the paths taken to plot the diagonal PSFs
    diagonal_paths = {'left_down': ['corner top left', 'top left', 'center', 'bottom right', 'corner bottom right'],
                      'left_up': ['corner bottom left', 'bottom left', 'center', 'top right', 'corner top right'],
                      'central': ['top', 'left', 'bottom', 'right', 'center']}

    filtered_half_images = [img for img in half_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera and img not in np.array(half_second_images)[washed_out_images]]
    filtered_five_images = [img for img in five_second_images if fits.getheader(img)['TCGAIN'] == gain and fits.getheader(img)['INSTRUME']==camera and img not in np.array(five_second_images)[washed_out_images]]

    # get indices of all the pics in the filtered images
    half_indexes = [half_second_images.index(img) for img in filtered_half_images]
    five_indexes = [five_second_images.index(img) for img in filtered_five_images]

    # determines the region that each of these images are in
    regions_halfsec = [assign_region(brights_halfsec[index]) for index in half_indexes]
    regions_fivesec = [assign_region(brights_5s[index]) for index in five_indexes]
    assert regions_halfsec==regions_fivesec, 'Regions are not the same!'

    diagonals_half, diagonals_five = [], []
    filtered_regions_halfsec, filtered_regions_fivesec = [],[]
    for i in np.arange(len(half_indexes)):
        if regions_halfsec[i] in diagonal_paths[diagonal_path]:
            diagonals_half.append(half_indexes[i])
            filtered_regions_halfsec.append(regions_halfsec[i])
        if regions_fivesec[i] in diagonal_paths[diagonal_path]:
            diagonals_five.append(five_indexes[i])
            filtered_regions_fivesec.append(regions_fivesec[i])

    assert diagonals_half == diagonals_five, 'Indices of 0.5s and 5s images do not match up'
    assert filtered_regions_halfsec == filtered_regions_fivesec, 'Filtered regions of 0.5s and 5s images do not match up'

    num_images = len(diagonals_half)

    # turns off the figure window if the plot command is False, used to take hybrids in diagonals only
    if plot == False:
        plt.ioff()
    fig = plt.figure(figsize=(9,7))

    # creates the plot with all the colorful regional rectangles that will appear in the diagonal slices
    sample_image_file = np.zeros([2004,2752])
    ax_regions = fig.add_subplot(2,1,1)
    ax_regions.imshow(sample_image_file,cmap='gray'), ax_regions.grid(False), ax_regions.set(xticks=[], yticks=[])

    for region in filtered_regions_fivesec:
        brightest_row, brightest_column = flipped_rectangle_centers[region]
        rect_color = region_colors[region]
        rectangle = patches.Rectangle((brightest_column-65,brightest_row-50), 130,100, edgecolor=rect_color,linewidth=1,facecolor='none')
        _= ax_regions.add_patch(rectangle)

    # before plotting images, we must determine the optimum subplot array
    if num_images <= 5:
        num_rows = 2
        num_cols = num_images
        subplot_number = num_cols + 1
    elif num_images > 5:
        num_rows = 4
        num_cols = num_images//2 # some floor division stuff
        subplot_number = 2*num_cols + 1

    # now plot all the images!
    hybrids = []
    for i in np.arange(num_images):
        hybrid = hybrid_PSF(diagonals_half[i])
        ax_zoomedimg = fig.add_subplot(num_rows, num_cols, subplot_number)
        ax_zoomedimg.grid(False), ax_zoomedimg.set(xticks=[], yticks=[])#,xlabel=[diagonals_half[i],diagonals_five[i]])
        ax_zoomedimg.imshow(hybrid,cmap='gray',vmin=0,vmax=1800)
        hybrids.append(hybrid)

        # changes spine color to match that of region
        spine_color = region_colors[filtered_regions_fivesec[i]]
        for border in ['top', 'bottom', 'left', 'right']:
            ax_zoomedimg.spines[border].set_color(spine_color)
            ax_zoomedimg.spines[border].set_linewidth(2)

        subplot_number +=1
    directions = {'left_up': 'Northeast', 'left_down': 'Southeast', 'central': 'Central'}
    # interesting bookend, why isn't mathrm being set by the above mpl.rcparam?
    ax_regions.set_title('Launch+18 Months\nPSF Direction: {0}'.format(directions[diagonal_path]), loc='center', fontsize=12)
    ax_regions.set_title('Camera: {}'.format(camera), loc='left', fontsize=12)
    ax_regions.set_title('Gain: {}'.format(gain), loc='right', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    if plot==False:
        plt.close(fig)
        plt.ion()
    return fig, [hybrids, filtered_regions_fivesec], [half_indexes, regions_halfsec]

def test_hybrid_plots():
    fig = plt.figure(figsize=(12,8))
    for a in np.arange(1,21):
        ax = fig.add_subplot(4,5,a)
        ax.imshow(hybrid_PSF(a),vmin=0,vmax=1800,cmap='gray')
        ax.grid(False)

'''with PdfPages('PSFs across FOV 18month.pdf') as pdf:
    plt.ioff()
    for gain in [8,10]:
        for camera in ['NCM','NFT']:
            figurine = FOV_hybrids(camera=camera, gain=gain)
            pdf.savefig(figurine,dpi=600)
            plt.close(figurine)
    plt.ion()'''

'''with PdfPages('Hybrid PSFs 18month.pdf') as pdf:
    plt.ioff()
    for gain in [8,10]:
        for camera in ['NFT', 'NCM']:
            for path in ['left_up', 'left_down', 'central']:
                figurine = visualize_PSFs(diagonal_path=path, camera=camera, gain=gain)[0]
                pdf.savefig(figurine)
                plt.close(figurine)
    plt.ion()'''

def calculate_MTF(PSF_image):
    '''Generates an MTF image based on the input PSF image.
    Verify this so that it works well, because I have to do some FFT and modulus and stuff.'''
    from scipy.fftpack import fft2
    complex_MTF = fft2(PSF_image.astype(float))
    MTF_real_only = np.abs(complex_MTF)
    normalized_MTF = MTF_real_only / np.max(MTF_real_only)
    return normalized_MTF

def test_MTF(list_of_indices):
    num_cols = len(list_of_indices)
    fig, (ax_up,ax_down) = plt.subplots(2, num_cols, figsize=(12,6))
    for i in np.arange(num_cols):
        ax_up[i].imshow(hybrid_PSF(list_of_indices[i]), cmap='gray')
        ax_down[i].imshow(MTF(hybrid_PSF(list_of_indices[i])), cmap='gray')
        ax_up[i].grid(False), ax_down[i].grid(False)
        ax_up[i].set(xticks=[],yticks=[],xlabel='Index: {}'.format(list_of_indices[i]), title='Hybrid PSF')
        ax_down[i].set(xticks=[],yticks=[],xlabel='Index: {}'.format(list_of_indices[i]), title='MTF')
    fig.tight_layout()
    return fig

def MTF_regions(diagonal='left_up', camera='NFT', gain=8, linestyle='o'):
    previous_fig, [hybrids, regions], misc = visualize_PSFs(diagonal, camera, gain, plot=False)
    plt.close(previous_fig)
    MTFs = [calculate_MTF(PSF) for PSF in hybrids]

    fig = plt.figure(figsize=(14,9))
    regions_ax = fig.add_subplot(2,1,1)
    regions_ax.imshow(np.zeros([2004, 2752]), cmap='gray')
    regions_ax.grid(False), regions_ax.set(xticks=[], yticks=[])

    num_rows = 2
    num_cols = 2
    subplot_number = num_cols + 1

    (ax1, ax2) = fig.add_subplot(num_rows,num_cols,subplot_number), fig.add_subplot(num_rows,num_cols,subplot_number+1)
    for i in np.arange(len(MTFs)):
        # picks out the horizontal and vertical component of the MTF starting from the origin (upper left corner)
        horizontal = MTFs[i][0, :5]
        vertical = MTFs[i][:5, 0]
        assert len(horizontal) == len(vertical)

        # sets the x-axis values properly in terms of spatial frequency
        spatial_frequencies = np.arange(len(horizontal)) / (10 * 2.2e-3) # n/(10 pixels * 2.2e-3mm) to get units of 1/mm

        region = regions[i]
        rect_color = region_colors[region]
        brightest_row, brightest_column = flipped_rectangle_centers[region]
        rectangle = patches.Rectangle((brightest_column-65,brightest_row-50), 130,100, edgecolor=rect_color,linewidth=1,facecolor='none')
        _= regions_ax.add_patch(rectangle)

        # plots the lines
        ax1.plot(spatial_frequencies, horizontal, color=rect_color, label=region)
        ax2.plot(spatial_frequencies, vertical, color=rect_color, label=region)
        # emphasizes points
        ax1.plot(spatial_frequencies, horizontal, linestyle, color=rect_color, label=region)
        ax2.plot(spatial_frequencies, vertical, linestyle, color=rect_color, label=region)

    #ax1.legend(), ax2.legend()
    ax1.set(xlabel='Spatial Frequency ' + r'$\mathrm{[mm^{-1}]}$', ylabel='MTF')
    ax2.set(xlabel='Spatial Frequency ' + r'$\mathrm{[mm^{-1}]}$', ylabel='MTF')
    ax1.set_title('Horizontal MTF', fontsize=12)
    ax2.set_title('Vertical MTF', fontsize=12)
    regions_ax.set_title('Camera: {}'.format(camera), loc='left')
    regions_ax.set_title('MTF Regions\nLaunch+18 Months\nDirection: {}'.format(directions[diagonal]), loc='center')
    regions_ax.set_title('Gain: {}'.format(gain), loc='right')
    fig.tight_layout()
    return fig

def save_MTFs(dpi=500):
    plt.ioff()
    for camera in ['NFT', 'NCM']:
        for gain in [8,10]:
            for direction in directions.keys():
                f = MTF_regions(diagonal=direction, camera=camera, gain=gain)
                f.savefig('MTF 18months/MTF Camera {} {} Gain {}'.format(camera,directions[direction],gain), dpi=dpi)
                plt.close(f)
    plt.ion()
