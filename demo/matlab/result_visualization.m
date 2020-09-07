%% ---- Monte carlo performanced visualization ---- %%
load('data/data.mat')
figure(1);


hold on;
plot(Nzs(1:10), nn_mse_over_n(1:10),'color','green','LineWidth',1.1);
plot(Nzs(1:10), tn_mse_over_n(1:10),'color','blue','LineWidth',1.1);
for ii = 1:size(cn_mse_over_n,1)
    plot(Nzs(1:10), cn_mse_over_n(ii,1:10),'color','black','LineWidth',ii*0.35);
end
plot(Nzs(1:6), dn_mse_over_n(1:6),'color','red', 'LineWidth',1.1);
xlim([5,300]);
xlabel('nodes');
ylabel('MSE');
set(gca,'Yscale','log')
set(gcf,'Position',[0 0 1000 800])




figure(2);


t = 1:99;
hold on;
plot(t, nn_mse_temp_o(t),'color','green','LineWidth',1.1);
plot(t, tn_mse_temp_o(t),'color','blue','LineWidth',1.1);
plot(t, dn_mse_temp_o(t),'color','red','LineWidth',1.1);
for ii = 1:size(cn_mse_over_n,1)
    plot(t, cn_mse_temp_o(ii,t),'color','black','LineWidth',ii*0.35);
end
xlim([1,99]);
ylim([0,3]);
xlabel('k');
ylabel('MSE');
set(gca,'Yscale','log')
set(gcf,'Position',[0 0 1000 800])


figure(3);
hold on;
plot(Nzs(1:6), dn_time_over_n(1:6),'color','red','LineWidth',1.1);
plot(Nzs(1:10), tn_time_over_n(1:10),'color','blue','LineWidth',1.1);
xlim([5,300]);
xlabel('nodes');
ylabel('time');
set(gcf,'Position',[0 0 1000 800])


%% ---- Network configuration visualization ---- %%
load('data/network_data.mat')
%node_pose = unifrnd(-10,10,100,2);
figure(4);
hold on;
plot(target_path(:,1), target_path(:,2),'color', 'red','LineWidth',1.1); 
plot(node_pose(:,1), node_pose(:,2),'.b', 'MarkerSize', 15); 
xlim([-10,10]);
ylim([-10,10]);
xlabel('x');
ylabel('y');
axis square;
set(gcf,'Position',[0 0 1000 800])


figure(5);
imagesc(A);
axis image;
set(gca, 'ydir', 'normal' )

c = colorbar('FontSize', 8);
ax = gca;
axpos = ax.Position;
cpos = c.Position;
cpos(3) = 0.5*cpos(3);
cpos(1) = 1.01*cpos(1);
c.Position = cpos;
ax.Position = axpos;


set(gcf,'Position',[0 0 1000 800])
