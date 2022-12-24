% image dir must contain only image files
obj_dir = 'D:\HUST\NGHIEN CUU\CH-VinIF\Paper dataset\Position-data\Matlab-export\Gray_image' ;
backgnd_dir = 'D:\HUST\NGHIEN CUU\CH-VinIF\Paper dataset\Position-data\Background\Gray_background';
anno_dir = 'D:\HUST\NGHIEN CUU\CH-VinIF\Paper dataset\Position-data\All_Label';
sync_dir = 'D:\HUST\NGHIEN CUU\CH-VinIF\Paper dataset\Position-data\Synthetic_dataset\Sync_test_img';
sync_anno = 'D:\HUST\NGHIEN CUU\CH-VinIF\Paper dataset\Position-data\Synthetic_dataset\Sync_test_anno';
low_default = 70; high_default = 95; reso_default = (high_default-low_default)/256;
ins_num = 2500;   % instance number

obj_list = dir(obj_dir);
backgnd_list = dir(backgnd_dir);
image_id = cell([1, ins_num]);
bbox = cell([1, ins_num]);
class = cell([1, ins_num]);
t = 0;
while t<ins_num
    t = t+1;
    %% import background and its range
    backgnd_idx = randi(length(backgnd_list)-2);
    backgnd_img = imread(fullfile(backgnd_dir, backgnd_list(2 + backgnd_idx).name)); % first-2 index are not image
    temp_min = range.min(backgnd_idx);  % *F
    temp_max = range.max(backgnd_idx);  % *F
    
    %% import objects
    obj_idx = randi(length(obj_list)-2);
    obj_path = fullfile(obj_dir, obj_list(2 + obj_idx).name); % first-2 index are not image
    obj_img = imread(obj_path); 
    [fPath, fName, fExt] = fileparts(obj_list(2 + obj_idx).name); % first-2 index are not image
    
    %% calibrate temperature range to associate with background
    reso_backgnd = (temp_max-temp_min)/256;
    obj_cal = obj_img;
    [h, w] = size(obj_cal);
    if (temp_min >= low_default) && (temp_min < high_default) && (temp_max >= high_default)
        for i=1:h
            for j=1:w
                if(obj_img(i,j)*reso_default + low_default < temp_min)
                    obj_cal(i,j) = 0;
                else
                    obj_cal(i,j) = (obj_img(i,j)*reso_default + low_default - temp_min)/reso_backgnd;
                end
            end
        end
    end
    if (temp_min < low_default) && (temp_max >= low_default) && (temp_max < high_default)
        for i=1:h
            for j=1:w
                if(obj_img(i,j)*reso_default + low_default >= temp_max)
                    obj_cal(i,j) = 255;
                else
                    obj_cal(i,j) = (obj_img(i,j)*reso_default + low_default - temp_min)/reso_backgnd;
                end
            end
        end
    end
    if (temp_min < low_default) && (temp_max >= high_default)
        for i=1:h
            for j=1:w
                obj_cal(i,j) = (obj_img(i,j)*reso_default + low_default - temp_min)/reso_backgnd;
            end
        end
    end
    if (temp_min >= low_default) && (temp_max < high_default)
        for i=1:h
            for j=1:w
                if (obj_img(i,j)*reso_default + low_default >= temp_max)
                    obj_cal(i,j) = 255;
                elseif (obj_img(i,j)*reso_default + low_default < temp_min)
                    obj_cal(i,j) = 0;
                else
                    obj_cal(i,j) = (obj_img(i,j)*reso_default + low_default - temp_min)/reso_backgnd;
                end
            end
        end
    end
    %% import annotation
    anno_path = fullfile(anno_dir, strcat(fName, '.txt'));
    if exist(anno_path, 'file')
        annomatric = readmatrix(anno_path);
    end
    
    %% get object
    idx = randi(size(annomatric, 1)); % random 1 object in anno file
    anno = annomatric(idx,:);
    [h, w] = size(obj_cal);
    xmin = ceil((anno(2) - anno(4)/2)*w);
    xmax = floor((anno(2) + anno(4)/2)*w);
    ymin = ceil((anno(3) - anno(5)/2)*h);
    ymax = floor((anno(3) + anno(5)/2)*h);
    object = obj_cal(ymin:ymax, xmin:xmax);
    
    %% scale object and creat new object image (obj_new)
    object_new = imresize(object, 0.5+rand());
    tform = randomAffine2d(XReflection=true);
    object_new = imwarp(object_new,tform);
    [h_new, w_new] = size(object_new);
    if h_new >= h-2 || w_new >= w-2
        t = t-1;
        continue;
    end
    
    xmin_new = randi(w-w_new-2)+1;
    ymin_new = randi(h-h_new-2)+1;
    
    obj_new = obj_cal;
    obj_new(ymin_new:ymin_new+h_new-1, xmin_new:xmin_new+w_new-1) = object_new;
    
    %% creart mask
    mask = false([h, w]);
    mask(ymin_new:ymin_new+h_new-3, xmin_new:xmin_new+w_new-3) = 1;
    
    
%     subplot(2,3,1)
%     imshow(obj_img)
%     subplot(2,3,2)
%     imshow(obj_cal)
%     subplot(2,3,3)
%     imshow(object_new)
%     subplot(2,3,4)
%     imshow(obj_new)
%     subplot(2,3,5)
%     imshow(backgnd_img)
%     subplot(2,3,6)
    im_out = PIE_modify(backgnd_img, obj_new, mask, 1, 1);
%     im_anno = cat(3, im_out, im_out, im_out);
%     im_anno(xmin_new:xmin_new+w_new-1,:) = [255,0,0];
    label = '';
    switch anno(1) 
        case 0
            label = 'creeping';
        case 1
            label = 'crawling';
        case 2
            label = 'stooping';
        case 3
            label = 'climbing'; 
        case 4            
            label = 'other';
    end
    im_anno = insertObjectAnnotation(im_out,'rectangle',[xmin_new,ymin_new,w_new-2,h_new-2],label);
%     set(gcf, 'Position', get(0, 'Screensize'));
%     pause(0.1);


    %% creat new sync image and corresponding bbox
    if t < 10
        ord = '0000' + string(t);
    elseif t < 100
        ord = '000' + string(t);
    elseif t < 1000
        ord = '00' + string(t);
    elseif t < 10000
        ord = '0' + string(t);
    else
        ord = string(t);
    end
    imwrite(im_out, fullfile(sync_dir, strcat(ord, '.BMP')));
    imwrite(im_anno, fullfile(sync_anno, strcat(ord, '.JPG')));
    image_id{t} = t;
    bbox{t} = [xmin_new,ymin_new,w_new-2,h_new-2];
    class{t} = anno(1);
end
json = jsonencode(struct('image_id', image_id, 'bbox', bbox, 'class', class));
f = fopen('sync_test_anno.json', 'w');
fprintf(f, json);
