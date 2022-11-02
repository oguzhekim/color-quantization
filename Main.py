
# return img, nested list
def read_ppm_file(f):
    fp = open(f)
    fp.readline()  # reads P3 (assume it is P3 file)
    lst = fp.read().split()
    n = 0
    n_cols = int(lst[n])
    n += 1
    n_rows = int(lst[n])
    n += 1
    max_color_value = int(lst[n])
    n += 1
    img = []
    for r in range(n_rows):
        img_row = []
        for c in range(n_cols):
            pixel_col = []
            for i in range(3):
                pixel_col.append(int(lst[n]))
                n += 1
            img_row.append(pixel_col)
        img.append(img_row)
    fp.close()
    return img, max_color_value


# Works
def img_printer(img):
    row = len(img)
    col = len(img[0])
    cha = len(img[0][0])
    for i in range(row):
        for j in range(col):
            for k in range(cha):
                print(img[i][j][k], end=" ")
            print("\t|", end=" ")
        print()


filename = input()
operation = int(input())


# DO_NOT_EDIT_ANYTHING_ABOVE_THIS_LINE
max_value = read_ppm_file(filename)[1]
img = read_ppm_file(filename)[0]
l = len(img)  # It's used in some operations.
min_value = 0


def op1(new_min, new_max):
    # Creates new image as an empty list
    new_img = []
    # Iterates rows
    for i in range(l):
        # Creates empty row to add to new image when row is filled
        new_row = []
        # Iterates pixels in a single row
        for j in range(l):
            # Creates empty pixel to add to new row when pixel is filled
            new_pixel = []
            # Iterates channels in a single pixel
            for k in range(3):
                # Stores the old channel value
                old_val = img[i][j][k]
                # Calculate the new channel value
                new_val = (old_val-min_value)*(new_max-new_min)/(max_value-min_value)+new_min
                # Adds new channel value to the pixel
                new_pixel.append(round(new_val, 4))
            # When all 3 of the new channel values are added to the pixel this means new pixel is ready to be appended to the new row
            new_row.append(new_pixel)
        # When entire row is filled, append that row to new image
        new_img.append(new_row)
    # When iteration is complete it means all the rows are appended to new image.
    return img_printer(new_img)


def op2():
    r = g = b = 0
    # This loop iterates every r values in the image and add the value to r variable defined above.
    # After iterating r values, same thing happens for g and b values.
    for i in range(3):
        for row_num in range(l):
            for col_num in range(l):
                if i == 0: r += img[row_num][col_num][i]  # i==0 Iterating over r values
                if i == 1: g += img[row_num][col_num][i]  # i==1 Iterating over g values
                if i == 2: b += img[row_num][col_num][i]  # i==2 Iterating over b values
    # Sum of every r, g, b values is determined. Now averages of the values is calculated below.
    ave_r = r/(l*l)
    ave_g = g/(l*l)
    ave_b = b/(l*l)
    numerator_r = numerator_g = numerator_b = 0  # Numerators of standard deviation values are defined here
    # This loop iterates every r values in the image and calculates and adds the value to numerator variables defined above.
    # After iterating r values, same thing happens for g and b values.
    for i in range(3):
        for row_num in range(l):
            for col_num in range(l):
                if i == 0: numerator_r += (img[row_num][col_num][i] - ave_r)**2  # i==0 Iterating over r values
                if i == 1: numerator_g += (img[row_num][col_num][i] - ave_g)**2  # i==1 Iterating over g values
                if i == 2: numerator_b += (img[row_num][col_num][i] - ave_b)**2  # i==2 Iterating over b values
    # Numerators of standard deviation values are calculated above.
    # Now standard deviations are calculated.
    s_deviation_r = (numerator_r/(l*l))**0.5 + 0.000001
    s_deviation_g = (numerator_g/(l*l))**0.5 + 0.000001
    s_deviation_b = (numerator_b/(l*l))**0.5 + 0.000001
    # Loop below iterates through every value of the image and apply z score normalization with averages and standard deviatons calculated above.
    for row_num in range(l):
        for col_num in range(l):
            for i in range(3):
                if i == 0:
                    img[row_num][col_num][i] = round((img[row_num][col_num][i] - ave_r)/s_deviation_r,4)
                if i == 1:
                    img[row_num][col_num][i] = round((img[row_num][col_num][i] - ave_g)/s_deviation_g,4)
                if i == 2:
                    img[row_num][col_num][i] = round((img[row_num][col_num][i] - ave_b)/s_deviation_b,4)
    return img_printer(img)


def op3():
    # Creates new image as an empty list
    new_img = []
    # Iterates rows
    for i in range(l):
        # Creates empty row to add to new image when row is filled
        new_row = []
        # Iterates pixels in a single row
        for j in range(l):
            # Takes the average of the channels in the pixel
            ave = sum(img[i][j])//3
            # Creates the new pixel
            new_pixel = [ave, ave, ave]
            # Adds new pixel to the row
            new_row.append(new_pixel)
        # When entire row is filled, append that row to new image
        new_img.append(new_row)
    # When iteration is complete it means all the rows are appended to new image.
    return img_printer(new_img)


# This function returns the filter, that will be used in op4 and op5, in a usable manner (3d matrix)
def read_filter(filter_name):
    # Read the filter file and make it a list
    filter_file = open(filter_name)
    lst = filter_file.read().split()
    # Get the length of that list
    m = int(len(lst)**0.5) # filter is mxm matrix
    # Creating a 3d matrix consists of zero's with the length of filter.
    matrix = [[[0, 0, 0] for i in range(m)] for j in range(m)]
    # Altering zero matrix with the values of filter
    for row_num in range(m):
        for col_num in range(m):
            for i in range(3): # Because there are 3 channels
                matrix[row_num][col_num][i] = float(lst[row_num*m+col_num])
    return matrix


def op4(image, kernel, stride):
    n = len(image)
    m = len(kernel)
    # Creates new image as an empty list
    new_img = []
    # In this operation I matched top left corner of filter to pixels in image.
    # This loop sets the row position that will be matched.
    # Row position that will be matched with top left corner of the filter can be n-m at most.
    # Stride just changes step of this loop.
    for row in range(0, n-m+1, stride):
        # Creates empty row to add to new image when row is filled
        new_row = []
        # This loop sets the column position that will be matched.
        # Column position that will be matched with top left corner of the filter can be n-m at most.
        # Stride just changes step of this loop.
        for col in range(0, n-m+1, stride):
            # At this point, top left corner of the filter is matched with necessary pixel of the image.
            # r,g,b values must reset when initial position change.
            r = g = b = 0
            # Loops below iterates mxm submatrix starting from the initial position.
            # x and y iterates the pixels of submatrix.
            # z is channel index.
            # So this loop iterates through r values of every pixel in mxm submatrix first.
            # While iterating, r values are multiplied with corresponding kernel value.
            # Result is added to the r=0 variable created above.
            # After that same process happens for g and b values.
            for z in range(3):
                for y in range(m):
                    for x in range(m):
                        channel_val = image[row + y][col + x][z]
                        if z == 0:  # this means its r value
                            r += kernel[y][x][z] * channel_val
                        elif z == 1:  # this means its g value
                            g += kernel[y][x][z] * channel_val
                        elif z == 2:  # this means its b value
                            b += kernel[y][x][z] * channel_val
            # Every r,g,b values of mxm submatrix is iterated and necessary calculations are made.
            # Calculated r,g,b channel values are added to the new pixel.
            new_pixel = [int(r), int(g), int(b)]
            # This loop checks if channel values are within the range. If not, it is rearranged.
            for c in range(3):
                if new_pixel[c] > max_value:
                    new_pixel[c] = max_value
                if new_pixel[c] < min_value:
                    new_pixel[c] = min_value
            # Adds new pixel to the row
            new_row.append(new_pixel)
        # When entire row is filled, append that row to new image
        new_img.append(new_row)
    # When iteration is complete it means all the rows are appended to new image.
    return img_printer(new_img)


def op5(kernel, stride):
    m = len(kernel)
    # In this operation padding depends on kernel and image size.
    # So I created a zero matrix with that size. (padded image size)
    matrix = [[[0,0,0] for i in range(l+m-1)] for j in range(l+m-1)]
    # Now I need to modify zero matrix by putting the values of original image to the correct positions.
    # Initial position that will be modified is calculated below.
    initial_pos = (len(matrix)-l)//2
    # In loop below I start with coordinate (initial_pos, initial_pos) and set every element of the zero matrix equal to image values
    # When this loop completes there will be pixels that's not iterated. Those pixels will remain 0. So padding will be completed.
    for row_num in range(l):
        for col_num in range(l):
            for i in range(3):
                matrix[row_num+initial_pos][col_num+initial_pos][i] = img[row_num][col_num][i]
    # After padding is completed it's same as op4 so I return op4 with padded matrix.
    return op4(matrix, kernel, stride)


# This function is used in op6 to decide whether two pixel will be made equal or not.
def pixel_compare(pixel1, pixel2, range):
    if min(int(pixel1[0]), int(pixel2[0])) + range <= max(int(pixel1[0]), int(pixel2[0])):
        return False
    if min(int(pixel1[1]), int(pixel2[1])) + range <= max(int(pixel1[1]), int(pixel2[1])):
        return False
    if min(int(pixel1[2]), int(pixel2[2])) + range <= max(int(pixel1[2]), int(pixel2[2])):
        return False
    return True


def op6(range, row=0, col=0, visited_pixels=[]):
    # Visited pixel's coordinates is stored as a tuple so they will not be visited again and cause infinite recursion
    visited_pixels.append((row, col))
    # Current pixel is (0,0) in the beginning

    # If condition below is true that means we are not currently on last row and pixel below is not visited.
    # So we can compare the current pixel with pixel below.
    if row != len(img)-1 and (row+1, col) not in visited_pixels:
        if pixel_compare(img[row][col], img[row+1][col], range):
            img[row+1][col] = img[row][col]  # If two pixels are within the range, pixel below will be made equal to pixel above
        return op6(range, row+1, col)  # Current pixel becomes pixel below.
    # If condition below is true that means we are not currently on first row and pixel above is not visited.
    # So we can compare the current pixel with pixel above.
    elif row != 0 and (row-1, col) not in visited_pixels:
        if pixel_compare(img[row][col], img[row-1][col], range):
            img[row-1][col] = img[row][col]  # If two pixels are within the range, pixel above will be made equal to pixel below
        return op6(range, row-1, col)  # Current pixel becomes pixel above.
    # If condition below is true that means we are not currently on last column and pixel on the right is not visited.
    # So we can compare the current pixel with pixel on the right.
    elif col != len(img)-1 and (row, col+1) not in visited_pixels:
        if pixel_compare(img[row][col], img[row][col+1], range):
            img[row][col+1] = img[row][col]  # If two pixels are within the range, pixel on the right will be made equal to pixel on the left
        return op6(range, row, col+1)  # Current pixel becomes pixel on the right.
    # If condition below is true that means we are not currently on first column and pixel on the left is not visited.
    # So we can compare the current pixel with pixel on the left.
    elif col != 0 and (row, col-1) not in visited_pixels:
        if pixel_compare(img[row][col], img[row][col-1], range):
            img[row][col-1] = img[row][col]  # If two pixels are within the range, pixel on the left will be made equal to pixel on the right
        return op6(range, row, col-1)  # Current pixel becomes pixel on the left.
    return img_printer(img)


# This function is used in op7 to decide whether two channels will be made equal or not.
def channel_compare(ch1, ch2, range):
    if min(int(ch1), int(ch2)) + range <= max(int(ch1), int(ch2)):
        return False
    return True


def op7(range, row=0, col=0, channel=0, visited=[]):
    # Visited channel's coordinates is stored as a tuple so they will not be visited again and cause infinite recursion
    visited.append((row, col, channel))
    # Current channel is (0,0,0) in the beginning

    # If condition below is true that means we are not currently on last row and channel of pixel below is not visited.
    # So we can compare the current pixel's channel with the channel of pixel below.
    if row != len(img)-1 and (row+1, col, channel) not in visited:
        if channel_compare(img[row][col][channel], img[row+1][col][channel], range):
            img[row+1][col][channel] = img[row][col][channel]  # If two channels are within the range, channel below will be made equal to channel above
        return op7(range, row+1, col, channel)  # Current channel becomes channel below.
    # If condition below is true that means we are not currently on first row and channel of pixel above is not visited.
    # So we can compare the current pixel's channel with the channel of pixel above.
    elif row != 0 and (row-1, col, channel) not in visited:
        if channel_compare(img[row][col][channel], img[row-1][col][channel], range):
            img[row-1][col][channel] = img[row][col][channel]  # If two channels are within the range, channel above will be made equal to channel below
        return op7(range, row-1, col, channel)  # Current channel becomes channel above.
    # If condition below is true that means we are not currently on last column and channel of pixel on the right is not visited.
    # So we can compare the current pixel's channel with the channel of pixel on the right.
    elif col != len(img)-1 and (row, col+1, channel) not in visited:
        if channel_compare(img[row][col][channel], img[row][col+1][channel], range):
            img[row][col+1][channel] = img[row][col][channel]  # If two channels are within the range, channel of pixel on the right will be made equal to channel of pixel on the left
        return op7(range, row, col+1, channel)  # Current channel becomes channel of pixel on the right.
    # If condition below is true that means we are not currently on first column and channel of pixel on the left is not visited.
    # So we can compare the current pixel's channel with the channel of pixel on the left.
    elif col != 0 and (row, col-1, channel) not in visited:
        if channel_compare(img[row][col][channel], img[row][col-1][channel], range):
            img[row][col-1][channel] = img[row][col][channel]  # If two channels are within the range, channel of pixel on the left will be made equal to channel of pixel on the right
        return op7(range, row, col-1, channel)  # Current channel becomes channel of pixel on the left.
    # If condition below is true that means we are not currently on last channel of a pixel and channel on the right of the same pixel is not visited.
    # So we can compare the current channel with the channel on the right of the same pixel.
    elif channel != 2 and (row, col, channel+1) not in visited:
        if channel_compare(img[row][col][channel], img[row][col][channel+1], range):
            img[row][col][channel+1] = img[row][col][channel]  # If two channels are within the range, right channel will be made equal to left channel of the same pixel.
        return op7(range, row, col, channel+1)  # Current channel becomes channel on the right of the same pixel.
    # If condition below is true that means we are not currently on first channel of a pixel and channel on the left of the same pixel is not visited.
    # So we can compare the current channel with the channel on the left of the same pixel.
    elif channel != 0 and (row, col, channel-1) not in visited:
        if channel_compare(img[row][col][channel], img[row][col][channel-1], range):
            img[row][col][channel-1] = img[row][col][channel]  # If two channels are within the range, left channel will be made equal to right channel of the same pixel.
        return op7(range, row, col, channel-1)  # Current channel becomes channel on the left of the same pixel.
    return img_printer(img)


if operation == 1:
    new_min = int(input())
    new_max = int(input())
    op1(new_min, new_max)
if operation == 2:
    op2()
if operation == 3:
    op3()
if operation == 4:
    filter_name = input()
    stride = int(input())
    kernel = read_filter(filter_name)
    op4(img, kernel, stride)
if operation == 5:
    filter_name = input()
    stride = int(input())
    kernel = read_filter(filter_name)
    op5(kernel, stride)
if operation == 6:
    r = int(input())
    op6(r)
if operation == 7:
    r = int(input())
    op7(r)


# DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE

