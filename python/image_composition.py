#!/usr/bin/env python3

import json
import warnings
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps

class MaskJsonUtils():
    """ Creates a JSON definition file for image masks.
    """

    def __init__(self, output_dir):
        """ Initializes the class.
        Args:
            output_dir: the directory where the definition file will be saved
        """
        self.output_dir = output_dir
        self.masks = dict()
        self.super_categories = dict()

    def add_category(self, category, super_category):
        """ Adds a new category to the set of the corresponding super_category
        Args:
            category: e.g. 'eagle'
            super_category: e.g. 'bird'
        Returns:
            True if successful, False if the category was already in the dictionary
        """
        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet, create a new set
            self.super_categories[super_category] = {category}
        elif category in self.super_categories[super_category]:
            # Category is already accounted for
            return False
        else:
            # Add the category to the existing super category set
            self.super_categories[super_category].add(category)

        return True # Addition was successful

    def add_mask(self, image_path, mask_path, color_categories):
        """ Takes an image path, its corresponding mask path, and its color categories,
            and adds it to the appropriate dictionaries
        Args:
            image_path: the relative path to the image, e.g. './images/00000001.png'
            mask_path: the relative path to the mask image, e.g. './masks/00000001.png'
            color_categories: the legend of color categories, for this particular mask,
                represented as an rgb-color keyed dictionary of category names and their super categories.
                (the color category associations are not assumed to be consistent across images)
        Returns:
            True if successful, False if the image was already in the dictionary
        """
        if self.masks.get(image_path):
            return False # image/mask is already in the dictionary

        # Create the mask definition
        mask = {
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

        # Regardless of color, we need to store each new category under its supercategory
        for _, item in color_categories.items():
            self.add_category(item['category'], item['super_category'])

        return True # Addition was successful

    def get_masks(self):
        """ Gets all masks that have been added
        """
        return self.masks

    def get_super_categories(self):
        """ Gets the dictionary of super categories for each category in a JSON
            serializable format
        Returns:
            A dictionary of lists of categories keyed on super_category
        """
        serializable_super_cats = dict()
        for super_cat, categories_set in self.super_categories.items():
            # Sets are not json serializable, so convert to list
            serializable_super_cats[super_cat] = list(categories_set)
        return serializable_super_cats

    def write_masks_to_json(self):
        """ Writes all masks and color categories to the output file path as JSON
        """
        # Serialize the masks and super categories dictionaries
        serializable_masks = self.get_masks()
        serializable_super_cats = self.get_super_categories()
        masks_obj = {
            'masks': serializable_masks,
            'super_categories': serializable_super_cats
        }

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'mask_definitions.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(masks_obj))

class ImageComposition():
    """ Composes images together in random ways, applying transformations to the foreground to create a synthetic
        combined image.
    """

    def __init__(self):
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8 # 00000027.png, supports up to 100 million images
        self.max_foregrounds = 1
        self.mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        assert len(self.mask_colors) >= self.max_foregrounds, 'length of mask_colors should be >= max_foregrounds'

    def _validate_and_process_args(self, args):
        # Validates input arguments and sets up class variables
        # Args:
        #     args: the ArgumentParser command line arguments

        self.silent = args.silent

        # Validate the count
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count

        # Validate the width and height
        assert args.width >= 64, 'width must be greater than 64'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64'
        self.height = args.height

        # Validate and process the output type
        if args.output_type is None:
            self.output_type = '.jpg' # default
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type}'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

        # Validate and process output and input directories
        self._validate_and_process_output_directory()
        self._validate_and_process_input_directory()

    def _validate_and_process_output_directory(self):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images'
        self.masks_output_dir = self.output_dir / 'masks'

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_output_dir.mkdir(exist_ok=True)
        self.masks_output_dir.mkdir(exist_ok=True)

        if not self.silent:
            # Check for existing contents in the images directory
            for _ in self.images_output_dir.iterdir():
                # We found something, check if the user wants to overwrite files or quit
                should_continue = input('output_dir is not empty, files may be overwritten.\nContinue (y/n)? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def _validate_and_process_input_directory(self):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'input_dir does not exist: {args.input_dir}'

        for x in self.input_dir.iterdir():
            if x.name == 'foregrounds':
                self.foregrounds_dir = x
            elif x.name == 'backgrounds':
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        # Validates input foregrounds and processes them into a foregrounds dictionary.
        # Expected directory structure:
        # + foregrounds_dir
        #     + super_category_dir
        #         + category_dir
        #             + foreground_image.png
        
        self.foregrounds_dict = dict()
        
        for super_category_dir in self.foregrounds_dir.iterdir():
            if not super_category_dir.is_dir():
                warnings.warn(f'file found in foregrounds directory (expected super-category directories), ignoring: {super_category_dir}')
                continue

            # This is a super category directory
            for category_dir in super_category_dir.iterdir():
                if not category_dir.is_dir():
                    warnings.warn(f'file found in super category directory (expected category directories), ignoring: {category_dir}')
                    continue
                masks_used = False
                # This is a category directory
                # Is there a masks directory?
                for image_file in category_dir.iterdir():
                    if image_file.is_dir():
                        if image_file.stem == 'masks':
                            masks_used = True
                            print('\'masks\' dir found, using pre-made foreground masks for class \'{}\''.format(image_file.parents[0].stem))
                            break
                    
                for image_file in category_dir.iterdir():
                    if not image_file.is_file():
                        if image_file.is_dir():
                            # Ignore completely, do not wark of masks dir
                            if image_file.stem == 'masks':
                                continue
                        warnings.warn(f'a directory was found inside a category directory, ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                        continue
                    
                    image_dict = {'image': image_file, 'mask': None}
                    if masks_used:
                        mask_file = image_file.parents[0] / 'masks' / image_file.name
                        if not mask_file.is_file():
                            warnings.warn(f'mask not found for {str(mask_file)}, skipping...')
                            continue
                        else:
                            image_dict['mask'] = mask_file
                    
                    # Valid foreground image, add to foregrounds_dict
                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()

                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_dict)
        
        assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'a directory was found inside the backgrounds directory, ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(f'background must match an accepted type {str(self.allowed_background_types)}, ignoring: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds were found'

    def _generate_images(self):
        # Generates a number of images and creates segmentation masks, then
        # saves a mask_definitions.json file that describes the dataset.

        print(f'Generating {self.count} images with masks...')

        mju = MaskJsonUtils(self.output_dir)

        # Create all images/masks (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # Randomly choose a background
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(1, self.max_foregrounds)
            foregrounds = []
            for fg_i in range(num_foregrounds):
                # Randomly choose a foreground
                super_category = random.choice(list(self.foregrounds_dict.keys()))
                category = random.choice(list(self.foregrounds_dict[super_category].keys()))
                foreground_dict = random.choice(self.foregrounds_dict[super_category][category])
                
                # Get the color
                mask_rgb_color = self.mask_colors[fg_i]

                foregrounds.append({
                    'super_category':super_category,
                    'category':category,
                    'foreground_dict':foreground_dict,
                    'mask_rgb_color':mask_rgb_color
                })

            # Compose foregrounds and background
            composite, mask = self._compose_images(foregrounds, background_path)

            # Create the file name (used for both composite and mask)
            save_filename = f'{i:0{self.zero_padding}}' # e.g. 00000023.jpg

            # Save composite image to the images sub-directory
            composite_filename = f'{save_filename}{self.output_type}' # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename # e.g. my_output_dir/images/00000023.jpg
            composite = composite.convert('RGB') # remove alpha
            composite.save(composite_path)

            # Save the mask image to the masks sub-directory
            mask_filename = f'{save_filename}.png' # masks are always png to avoid lossy compression
            mask_path = self.output_dir / 'masks' / mask_filename # e.g. my_output_dir/masks/00000023.png
            mask.save(mask_path)

            color_categories = dict()
            for fg in foregrounds:
                # Add category and color info
                mju.add_category(fg['category'], fg['super_category'])
                color_categories[str(fg['mask_rgb_color'])] = \
                    {
                        'category':fg['category'],
                        'super_category':fg['super_category']
                    }
            
            # Add the mask to MaskJsonUtils
            mju.add_mask(
                composite_path.relative_to(self.output_dir).as_posix(),
                mask_path.relative_to(self.output_dir).as_posix(),
                color_categories
            )

        #Write masks to json
        mju.write_masks_to_json()

    def _compose_images(self, foregrounds, background_path):
        # Composes a foreground image and a background image and creates a segmentation mask
        # using the specified color. Validation should already be done by now.
        # Args:
        #     foregrounds: a list of dicts with format:
        #       [{
        #           'super_category':super_category,
        #           'category':category,
        #           'foreground_path':foreground_path,
        #           'mask_rgb_color':mask_rgb_color
        #       },...]
        #     background_path: the path to a valid background image
        # Returns:
        #     composite: the composed image
        #     mask: the mask image
        
        def get_alpha(image_in, size=None, paste_position=None):
            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            alpha_mask = image_in.getchannel(3)
            if size == None:
                new_size = image_in.size
            else:
                new_size = size
            new_alpha_mask = Image.new('L', new_size, color = 0)
            new_alpha_mask.paste(alpha_mask, paste_position)
            return new_alpha_mask
        
        # Open background and convert to RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # Crop background to desired size (self.width x self.height), randomly positioned
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - self.width
        max_crop_y_pos = bg_height - self.height
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))
        composite_mask = Image.new('RGB', composite.size, 0)

        for fg in foregrounds:
            
            fg_dict = fg['foreground_dict']
            fg_path = fg_dict['image']
            fg_mask_path = fg_dict['mask']
            
            # Is a seperate mask file used (not alpha channel)?
            if fg_mask_path == None:
                uses_mask = False
            else:
                uses_mask = True
                
            
             # ** Apply Transformations **
            rot_jitter = 10  # degrees
            scale_range = (0.9, 1.1)  # percent
            bright_range = (.7, 1.1)
            pos_jitter = (20, 10)  # pixels
            
            t_params = dict()
        
            # Get center point
            # Get monocrome image from fg alpha, non-white = mass
            pretransform_alpha_image = Image.open(fg_path)
            pretransform_alpha_mask = get_alpha(pretransform_alpha_image)
            pretransform_alpha_mask_threshold = 127
            pretransform_alpha_thresh = pretransform_alpha_mask.point(lambda p:p > pretransform_alpha_mask_threshold and 255)
            pretransform_alpha_thresh = ImageOps.invert(pretransform_alpha_thresh)
            pretransform_alpha_thresh = pretransform_alpha_thresh.convert(mode='RGB')
            # Get center of mass (rotation center)
            mass = np.sum(np.asarray(pretransform_alpha_thresh), -1) < 255*3
            mass = mass / np.sum(np.sum(mass))
            mass_center_dx = np.sum(mass, 0)
            mass_center_dy = np.sum(mass, 1)
            t_params['center'] = dict()
            t_params['center']['X'] = np.sum(mass_center_dx * np.arange(pretransform_alpha_thresh.size[0]))
            t_params['center']['Y'] = np.sum(mass_center_dy * np.arange(pretransform_alpha_thresh.size[1]))
            
            # Add rotation jitter
            t_params['rotation'] = random.randint(-rot_jitter, rot_jitter)

            # Scale the foreground
            t_params['scale'] = random.uniform(scale_range[0], scale_range[1])

            # Adjust foreground brightness
            t_params['brightness'] = random.uniform(bright_range[0], bright_range[1])
            
            # Add some positional jitter
            t_params['translate'] = dict()
            t_params['translate']['X'] = random.randint(-pos_jitter[0], pos_jitter[0])
            t_params['translate']['Y'] = random.randint(-pos_jitter[1], pos_jitter[1])
            
            # Randomly decide if either axis should be flipped
            t_params['flip'] = dict()
            t_params['flip']['X'] = random.choice((True, False))
            t_params['flip']['Y'] = random.choice((True, False))
            
            # Add any other transformations here...
            
            
            # Perform transformations
            fg_image = self._transform_foreground(fg, fg_path, t_params, is_mask=False)
            if uses_mask:
                fg_image_mask = self._transform_foreground(fg, fg_mask_path, t_params, is_mask=True)

            # Checks (is for kids)
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, \
            f'foreground {fg_path} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({self.width}x{self.height}), check your input parameters'

            # Create a new foreground image as large as the composite and paste it on top
            new_fg_image = Image.new('RGBA', composite.size, color = (0, 0, 0, 0))
            new_fg_image.paste(fg_image)
            
            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            new_alpha_mask = get_alpha(fg_image, size=composite.size)
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)
            
            # Extract mask alpha, if applicable
            if uses_mask:
                mask_alpha_mask = fg_image_mask.getchannel(3)
                new_mask_alpha_mask = Image.new('L', composite.size, color = 0)
                new_mask_alpha_mask.paste(mask_alpha_mask)
            
            # Grab the alpha pixels above a specified threshold
            alpha_threshold = 200
            if uses_mask:
                alpha_mask_to_use = new_mask_alpha_mask
            else:
                alpha_mask_to_use = new_alpha_mask
            mask_arr = np.array(np.greater(np.array(alpha_mask_to_use), alpha_threshold), dtype=np.uint8)
            uint8_mask = np.uint8(mask_arr) # This is composed of 1s and 0s

            # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the mask
            mask_rgb_color = fg['mask_rgb_color']
            red_channel = uint8_mask * mask_rgb_color[0]
            green_channel = uint8_mask * mask_rgb_color[1]
            blue_channel = uint8_mask * mask_rgb_color[2]
            rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
            isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
            isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

            composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

        return composite, composite_mask
    
    def _transform_image_from_parameters(self, image_in, params_in, is_mask=False):
        image_out = image_in.copy()
       
        def check_for_param(p_in):
            if p_in in params_in:
                if params_in[p_in] != None:
                    return True
            return False
        
        # Flip image if applicable
        if check_for_param('flip'):
            if params_in['flip']['X']:
                image_out = ImageOps.mirror(image_out)
            if params_in['flip']['Y']:
                image_out = ImageOps.flip(image_out)
                    
        # Transform center to real center (if not None)
        if check_for_param('center'):
            w = image_out.size[0]
            h = image_out.size[1]
            # Check for flips and adjust centers if needed
            if check_for_param('flip'):
                if params_in['flip']['X']:
                    x = w - params_in['center']['X']
                else:
                    x = params_in['center']['X']
                if params_in['flip']['Y']:
                    y = h - params_in['center']['Y']
                else:
                    y = params_in['center']['Y']
            else:
                x = params_in['center']['X']
                y = params_in['center']['Y']
            dx = (w / 2) - (w - x)
            dy = (h / 2) - (h - y) 
            image_out = image_out.transform(image_out.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
        
        # Do Transforms
        if check_for_param('scale'):
            new_size = (int(image_out.size[0] * params_in['scale']), int(image_out.size[1] * params_in['scale']))
            image_out = image_out.resize(new_size, resample=Image.BICUBIC)
        if check_for_param('rotation'):
            image_out = image_out.rotate(params_in['rotation'], resample=Image.BICUBIC, expand=True)
        if check_for_param('brightness'):
            if not is_mask:
                enhancer = ImageEnhance.Brightness(image_out)
                image_out = enhancer.enhance(params_in['brightness'])
        
        
        
        # Add any other transformations here...
        
        # Adjust final dx and dy by translate
        final_dx = dx
        final_dy = dy
        if check_for_param('translate'):
            final_dx += params_in['translate']['X']
            final_dy += params_in['translate']['Y']
        
        # Transform back to original position
        if check_for_param('center'):
            image_out = image_out.transform(image_out.size, Image.AFFINE, (1, 0, -final_dx, 0, 1, -final_dy))
        
        image_out = image_out.crop((0, 0, image_in.size[0], image_in.size[1]))
        return image_out
    
    def _transform_foreground(self, fg, fg_path, params_in, is_mask=False):
        # Open foreground and get the alpha channel
        fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        assert np.any(fg_alpha == 0), f'foreground needs to have some transparency: {str(fg_path)}'
        
        fg_image = self._transform_image_from_parameters(fg_image, params_in, is_mask)
        return fg_image

    def _create_info(self):
        # A convenience wizard for automatically creating dataset info
        # The user can always modify the resulting .json manually if needed

        if self.silent:
            # No user wizard in silent mode
            return

        should_continue = input('Would you like to create dataset info json? (y/n) ').lower()
        if should_continue != 'y' and should_continue != 'yes':
            print('No problem. You can always create the json manually.')
            quit()

        print('Note: you can always modify the json manually if you need to update this.')
        info = dict()
        info['description'] = input('Description: ')
        info['url'] = input('URL: ')
        info['version'] = input('Version: ')
        info['contributor'] = input('Contributor: ')
        now = datetime.now()
        info['year'] = now.year
        info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

        image_license = dict()
        image_license['id'] = 0

        should_add_license = input('Add an image license? (y/n) ').lower()
        if should_add_license != 'y' and should_add_license != 'yes':
            image_license['url'] = ''
            image_license['name'] = 'None'
        else:
            image_license['name'] = input('License name: ')
            image_license['url'] = input('License URL: ')

        dataset_info = dict()
        dataset_info['info'] = info
        dataset_info['license'] = image_license

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'dataset_info.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(dataset_info))

        print('Successfully created {output_file_path}')


    # Start here
    def main(self, args):
        self._validate_and_process_args(args)
        self._generate_images()
        self._create_info()
        print('Image composition completed.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="The input directory. \
                        This contains a 'backgrounds' directory of pngs or jpgs, and a 'foregrounds' directory which \
                        contains supercategory directories (e.g. 'animal', 'vehicle'), each of which contain category \
                        directories (e.g. 'horse', 'bear'). Each category directory contains png images of that item on a \
                        transparent background (e.g. a grizzly bear on a transparent background).")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="The directory where images, masks, \
                        and json files will be placed")
    parser.add_argument("--count", type=int, dest="count", required=True, help="number of composed images to create")
    parser.add_argument("--width", type=int, dest="width", required=True, help="output image pixel width")
    parser.add_argument("--height", type=int, dest="height", required=True, help="output image pixel height")
    parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg (default)")
    parser.add_argument("--silent", action='store_true', help="silent mode; doesn't prompt the user for input, \
                        automatically overwrites files")

    args = parser.parse_args()

    image_comp = ImageComposition()
    image_comp.main(args)