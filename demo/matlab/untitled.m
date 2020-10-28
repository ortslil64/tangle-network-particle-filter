load('data/data1000.mat');
cn_mse = squeeze(CN_mse);
tn_mse = squeeze(mean(TN_mse,4));
dn_mse = squeeze(mean(DN_mse,4));
nn_mse = squeeze(mean(NN_mse,4));

figure(1);
h = zeros(1,4);
hold on;
plot(prctile(cn_mse,10)','k','LineWidth',0.5);
plot(prctile(cn_mse,25)','k','LineWidth',0.8);
h(1) = plot(prctile(cn_mse,50)','k','LineWidth',1, 'DisplayName','CPF');
plot(prctile(cn_mse,75)','k','LineWidth',1.5);
plot(prctile(cn_mse,90)','k','LineWidth',2);

plot(prctile(tn_mse,10)','b','LineWidth',0.5);
plot(prctile(tn_mse,25)','b','LineWidth',0.8);
h(2) = plot(prctile(tn_mse,50)','b','LineWidth',1,'DisplayName','TSN');
plot(prctile(tn_mse,75)','b','LineWidth',1.5);
plot(prctile(tn_mse,90)','b','LineWidth',2);

plot(prctile(nn_mse,10)','g','LineWidth',0.5);
plot(prctile(nn_mse,25)','g','LineWidth',0.8);
h(3) = plot(prctile(nn_mse,50)','g','LineWidth',1,'DisplayName','noninteracting');
plot(prctile(nn_mse,75)','g','LineWidth',1.5);
plot(prctile(nn_mse,90)','g','LineWidth',2);

plot(smooth(prctile(dn_mse(:,1:29),10)',5,'rlowess'),'r','LineWidth',0.5);
plot(smooth(prctile(dn_mse(:,1:29),25)',5,'rlowess'),'r','LineWidth',0.8);
h(4) = plot(smooth(prctile(dn_mse(:,1:29),50)',5,'rlowess'),'r','LineWidth',1,'DisplayName','DPF');
plot(smooth(prctile(dn_mse(:,1:29),75)',5,'rlowess'),'r','LineWidth',1.5);
plot(smooth(prctile(dn_mse(:,1:29),90)',5,'rlowess'),'r','LineWidth',2);
legend(h);
xlim([1,29]);
xlabel('k');
ylabel('MSE');