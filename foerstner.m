function [ fcoors ] = foerstner( img, coe )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    addpath('foerstnerOperator_v1_0') 
	[win, corner, circ, ~]=ip_fop( img, ...
                                                   ... intensity image (one channel, grey-level image)
            'DETECTION_METHOD',        'foerstner',  ... method for optimal search window: 'foerstner' (default) or 'koethe'   
            'SIGMA_N'                  ,coe.sigma_n,         ... standard deviation of (constant) image noise (default: 2.0)
            'DERIVATIVE_FILTER'        ,'gaussian2d',... filter for gradient: 'gaussian2d'(default) oder 'gaussian1d'
            'INTEGRATION_FILTER'       ,'gaussian',  ... integration kernel: 'box_filt' (default) oder 'gaussian' 
            'SIGMA_DERIVATIVE_FILTER'  ,coe.sigma_deriv,         ... size of derivative filter (sigma) (default: 1.0)
            'SIGMA_INTEGRATION_FILTER' ,coe.sigma_int,           ... size of integration filter (default: 1.41 * SIGMA_DIFF)
            'PRECISION_THRESHOLD'      ,coe.precision_thresh,         ... threshold for precision of points (default: 0.5 Pixel)    
            'ROUNDNESS_THRESHOLD'      ,coe.round_thresh,         ... threshold for roundness (default: 0.3)
            'SIGNIFICANCE_LEVEL'       ,coe.significance_level,       ... significance level for point classification (default: 0.999)
            'VISUALIZATION'            ,coe.visualize);       ... visualization on or off (default : 'off')
	
    fcoors = zeros(length(corner)+length(circ),2);
	%fprintf('%d windows detected: %d corners, %d circles\n', length(win), length(corner), length(circ));
        %{
        for fcoor_i=1:length(win)
           r = win(fcoor_i).r;
           c = win(lala).c;
           fcoors(fcoor_i,:) = [c,r];
        end
        %}
	fcoor_i = 1;
	for coor_i = 1:length(corner)
        r = corner(coor_i).r;
        c = corner(coor_i).c;
        fcoors(fcoor_i,:) = [floor(c),floor(r)];
        fcoor_i = fcoor_i + 1;
    end
	for coor_i = 1:length(circ)
        r = circ(coor_i).r;
        c = circ(coor_i).c;
        fcoors(fcoor_i,:) = [floor(c),floor(r)];
        fcoor_i = fcoor_i + 1;
	end

end

