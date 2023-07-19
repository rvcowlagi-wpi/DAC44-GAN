%{
PROGRAM DESCRIPTION
-------------------
This script verifies whether the generated samples are part of the set of true samples.
It calculates the maximum Euclidean distance between feature vectors of real data. If this 
distance is smaller than the Euclidean distance between the generated feature and any real 
feature vector, it indicates that the generated samples belong to the real data distribution.
%}


% Specify the file path of the .txt file
file_path = 'C:\Users\nubapat\Data\data_mintime_7_10000.txt';
file_path1 = 'C:\Users\nubapat\Data\y_pred.csv';

% Converts file into a matrix
R = readmatrix(file_path);
G = readmatrix(file_path1);
R = R';
G = G';
G = G(2:end, :);
z = zeros(size(R,1),size(R,1));
count = 0;

% Calculates max euclidean distance between real data
for i = 1:size(R,1)
    for j = 1:size(R,1)
     d = norm(R(:,i) - R(:,j));
     z(i,j) = d;
    end
end    
max_value = max(z(:)); 

% Calculates euclidean distance between real and generated data
for k = 1: size(G, 2)
    for n = 1:size(G,1)
        d1 = norm(G(:,k)-R(:,n));
        if d1 > max_value
            count = count + 1; % count = 0 at the end implies satisfaction of criterion
        else
            continue
        end
    end
end    