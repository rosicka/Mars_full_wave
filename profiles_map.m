clear all;close all;clc
jg=181;
wg=91;
hg=350;
%hg=120;
jing=linspace(-pi,pi,jg);
wei=linspace(-pi/2,pi/2,wg);
rm=3393;
d='mars\magnetic_profiles_2';
Bping = zeros(hg,jg, wg, 3); % Preallocate Bping for efficiency
imax=5;
jmax=10;
for i=1:jg
    for j=1:wg
        for k=1:hg
            r=rm+k;
            [x,y,z] = sph2cart(jing(i),wei(j),r);
            Bping(k,i,j,:)=mgao_r([x y z]','g_110_mm_q','h_110_mm_q');
            
            
        end
    end
end
% Process the magnetic field data for each height level 
for i=1:jg
    for j=1:wg
        bt=Bping(:,i,j,:);
        %hb=[btz(:,90:179) btz(:,1:89)];
        lat=-92+2*j;
        lon=-182+2*i;
        filename = sprintf('lat_%d_lon_%d.csv',lat,lon); 
        writematrix(bt,fullfile(d,filename))
    end
end

