img = im2double(imread('../input/synthesis/bricks.jpg'));

mag = 2;
width = 50;
overlap = 17;

inp_texture = rgb2gray(img);
[m,n] = size(inp_texture);
m_new = ceil(m*mag/width)*width+overlap;
n_new = ceil(n*mag/width)*width+overlap;
out_texture = zeros(m_new,n_new);
out_img = zeros(m_new,n_new,3);
block_len = width+overlap;

%intialisation
out_texture(1:block_len,1:block_len) = inp_texture(1:block_len,1:block_len);
out_img(1:block_len,1:block_len,:) = img(1:block_len,1:block_len,:); 

%for first row
i = 1;
mask = zeros(block_len,block_len);
mask(:,1:overlap)=1;
for j = 2:ceil(n*mag/width)
    new_block = out_texture((i-1)*width+1:(i-1)*width+block_len,(j-1)*width+1:(j-1)*width+block_len);
    sq_input = conv2(inp_texture.*inp_texture,rot90(mask,2),'valid');
    last_part = 2*conv2(inp_texture,rot90(new_block,2),'valid');
    errors = sum(new_block.*new_block,'all')+sq_input-last_part;
    mini = abs(min(errors,[],'all'));
    [x,y] = find(errors <= mini*1.1);
    rand_num = randi([1 length(x)],1);
    updated_block = inp_texture(x(rand_num):x(rand_num)+block_len-1,y(rand_num):y(rand_num)+block_len-1);
    newout_part = img(x(rand_num):x(rand_num)+block_len-1,y(rand_num):y(rand_num)+block_len-1,:);
    
    e_vals=(new_block-double(updated_block)).^2;
    e_vals=e_vals(:,1:overlap);
    E_vals=zeros(width+overlap,overlap);
    E_vals(1,:)=e_vals(1,:);
    argmins=zeros(width+overlap,overlap);
    path_chosen=zeros(width+overlap,overlap);
    for i1 = 2:block_len
        tempor_1=zeros(3,overlap);
        tempor_1(1,:)=[inf,E_vals(i1-1,1:overlap-1)];
        tempor_1(2,:)=E_vals(i1-1,:);
        tempor_1(3,:)=[E_vals(i1-1,2:overlap),inf];
        tempor_1=tempor_1+e_vals(i1,:);
        [E_vals(i1,:),argmins(i1,:)]=min(tempor_1);
    end

    path_chosen(argmins==2)=0;
    path_chosen(argmins==1)=-1;
    path_chosen(argmins==3)=1;
    bound=zeros(block_len,block_len);
    [useless1,min_index]=min(E_vals(block_len,:));
    bound(block_len,1:min_index)=1;
    for i2=block_len-1:-1:1
        bound(i2,1:min_index+path_chosen(i2+1,min_index))=1;
        min_index=min_index+path_chosen(i2+1,min_index);
    end

    img_bound = repmat(bound,[1 1 3]);
    final_block = new_block.*bound + double(updated_block).*(1-bound);
    out_texture((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap) = final_block;
    out_img((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap,:) = out_img((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap,:).*(img_bound)+double(newout_part).*(1-img_bound);
end

%for first column
j = 1;
mask = zeros(block_len,block_len);
mask(1:overlap,:)=1;
for i = 2:ceil(m*mag/width)
    new_block = out_texture((i-1)*width+1:(i-1)*width+block_len,(j-1)*width+1:(j-1)*width+block_len);
    sq_input = conv2(inp_texture.*inp_texture,rot90(mask,2),'valid');
    last_part = 2*conv2(inp_texture,rot90(new_block,2),'valid');
    errors = sum(new_block.*new_block,'all')+sq_input-last_part;
    mini = abs(min(errors,[],'all'));
    [x,y] = find(errors <= mini*1.1);
    rand_num = randi([1 length(x)],1);
    updated_block = inp_texture(x(rand_num):x(rand_num)+block_len-1,y(rand_num):y(rand_num)+block_len-1);
    newout_part = img(x(rand_num):x(rand_num)+block_len-1,y(rand_num):y(rand_num)+block_len-1,:);

    yes1=(double(updated_block)-new_block).^2;
    %e_vals is the value which stores the square of difference
    %between two overlap patches.
    e_vals=transpose(yes1(1:overlap,:));
    %E_vals stores the min error boundary cut penalty values
    E_vals=zeros(width+overlap,overlap);
    E_vals(1,:)=e_vals(1,:);
    %argmins gives the index that we choose at each point
    %path_chosen consists of information of where to travle from
    %the rpesent pixel i.e either left or right or bottom.
    argmins=zeros(width+overlap,overlap);
    path_chosen=zeros(width+overlap,overlap);
    for i1 = 2:block_len
        tempor_1=zeros(3,overlap);
        tempor_1(1,:)=[inf,e_vals(i1-1,1:overlap-1)];
        tempor_1(2,:)=e_vals(i1-1,:);
        tempor_1(3,:)=[e_vals(i1-1,2:overlap),inf];
        tempor_1=tempor_1+e_vals(i1,:);
        [E_vals(i1,:),argmins(i1,:)]=min(tempor_1);
    end

    path_chosen(argmins==2)=0;
    path_chosen(argmins==1)=-1;
    path_chosen(argmins==3)=1;
    %bound stores information about which pixels belong to the
    %present adding patch.
    bound=zeros(block_len,block_len);
    bound1=zeros(block_len,overlap);
    [useless1,min_index]=min(E_vals(block_len,:));
    bound1(block_len,1:min_index)=1;
    for i2=block_len-1:-1:1
        bound1(i2,1:min_index+path_chosen(i2+1,min_index))=1;
        min_index=min_index+path_chosen(i2+1,min_index);
    end
    bound(1:overlap,:)=transpose(bound1);

    img_bound = repmat(bound,[1 1 3]);
    final_block = new_block.*bound + double(updated_block).*(1-bound);
    out_texture((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap) = final_block;
    out_img((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap,:) = out_img((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap,:).*(img_bound)+double(newout_part).*(1-img_bound);
end

%remaining image
mask = zeros(block_len,block_len);            
mask(1:overlap,:)=1;
mask(:,1:overlap)=1;
for i = 2:ceil(m*mag/width)
    for j = 2:ceil(n*mag/width)
        new_block = out_texture((i-1)*width+1:(i-1)*width+block_len,(j-1)*width+1:(j-1)*width+block_len);
        sq_input = conv2(inp_texture.*inp_texture,rot90(mask,2),'valid');
        last_part = 2*conv2(inp_texture,rot90(new_block,2),'valid');
        errors = sum(new_block.*new_block,'all')+sq_input-last_part;
        mini = abs(min(errors,[],'all'));
        [x,y] = find(errors <= mini*1.1);
        rand_num = randi([1 length(x)],1);
        updated_block = inp_texture(x(rand_num):x(rand_num)+block_len-1,y(rand_num):y(rand_num)+block_len-1);
        newout_part = img(x(rand_num):x(rand_num)+block_len-1,y(rand_num):y(rand_num)+block_len-1,:);

        yes1=(double(updated_block)-new_block).^2;
        e_vals1=yes1(:,1:overlap);
        E_vals1=zeros(block_len,overlap);
        E_vals1(block_len,:)=e_vals1(block_len,:);
        argmins1=zeros(width+overlap,overlap);
        path_chosen1=zeros(width+overlap,overlap);
        for i1 = block_len-1:-1:1
            tempor_1=zeros(3,overlap);
            tempor_1(1,:)=[inf,e_vals1(i1+1,1:overlap-1)];
            tempor_1(2,:)=e_vals1(i1+1,:);
            tempor_1(3,:)=[e_vals1(i1+1,2:overlap),inf];
            tempor_1=tempor_1+e_vals1(i1,:);
            [E_vals1(i1,:),argmins1(i1,:)]=min(tempor_1);
        end

        path_chosen1(argmins1==2)=0;
        path_chosen1(argmins1==1)=-1;
        path_chosen1(argmins1==3)=1;
        e_vals2=transpose(yes1(1:overlap,:));
        E_vals2=zeros(block_len,overlap);
        E_vals2(block_len,:)=e_vals2(block_len,:);
        argmins2=zeros(width+overlap,overlap);
        path_chosen2=zeros(width+overlap,overlap);
        for i1 = block_len-1:-1:1
            tempor_1=zeros(3,overlap);
            tempor_1(1,:)=[inf,e_vals2(i1+1,1:overlap-1)];
            tempor_1(2,:)=e_vals2(i1+1,:);
            tempor_1(3,:)=[e_vals2(i1+1,2:overlap),inf];
            tempor_1=tempor_1+e_vals2(i1,:);
            [E_vals2(i1,:),argmins2(i1,:)]=min(tempor_1);
        end

        path_chosen2(argmins2==2)=0;
        path_chosen2(argmins2==1)=-1;
        path_chosen2(argmins2==3)=1;
        E_vals_com=E_vals1(1:overlap,:)+E_vals2(1:overlap,:);
        [useless1,ind_val]=min(diag(E_vals_com));
        bound=zeros(block_len,block_len);
        bound(1:ind_val,1:ind_val)=1;
        min_index=ind_val;
        for i2=ind_val+1:block_len
            bound(i2,1:min_index+path_chosen1(i2-1,min_index))=1;
            min_index=min_index+path_chosen1(i2-1,min_index);
        end
        min_index=ind_val;
        for i2=ind_val+1:block_len
            bound(1:min_index+path_chosen1(i2-1,min_index),i2)=1;
            min_index=min_index+path_chosen2(i2-1,min_index);
        end

        img_bound = repmat(bound,[1 1 3]);
        final_block = new_block.*bound + double(updated_block).*(1-bound);
        out_texture((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap) = final_block;
        out_img((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap,:) = out_img((i-1)*width+1:i*width+overlap,(j-1)*width+1:j*width+overlap,:).*(img_bound)+double(newout_part).*(1-img_bound);
    
    end
end

imshow(img); truesize; figure
final_img = out_img(1:m_new-overlap,1:n_new-overlap,:);
imshow(final_img);