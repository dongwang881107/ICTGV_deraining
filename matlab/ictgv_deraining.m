function img_derained = ictgv_deraining(img_y, params)
% Multi-directional rain streak removal using infimal convolution of oscillation TGV 
% Dong Wang, 2021-07-07

% Unzip the parameters
    num_d = params.num_d;

    [m,n] = size(img_y);
     w = zeros(m,n,2);
     n = zeros(m,n,num_d);
     w_t = zeros(m,n,num_d,2);
     
     p = zeros(m,n,2);
     q = zeros(m,n,3);
     p_t = zeros(m,n,num_d,2);
     q_t = zeros(m,n,num_d,3);
    


    img_derained = 1-img_y;

end