load('data.mat')
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

figure(2);
hold on;
plot(Nzs(1:6), dn_time_over_n(1:6),'color','red','LineWidth',1.1);
plot(Nzs(1:10), tn_time_over_n(1:10),'color','blue','LineWidth',1.1);
xlim([5,300]);
xlabel('nodes');
ylabel('time');

figure(3);
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
set(gca,'Yscale','linear')
