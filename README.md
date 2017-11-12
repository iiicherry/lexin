# lexin 预测客户逾期
https://www.nowcoder.com/activity/lexin2017/index  <br>
时间原因，并未来得及提交答案，但是希望是一个好的开始~  <br>

## 数据介绍：
### 用户信息（userdata）
fuid_md5：MD5加密后用户ID <br>
fschoolarea_name_md5：MD5加密后学校完整名字 <br>
fage：注册时点的年龄 <br>
fsex：性别 <br>
fis_entrance_exam：是否统招 <br>
fregister_time：注册时间 <br>
fpocket_auth_time：授信时间 <br>
fdomicile_provice：籍贯省份 <br>
fdomicile_city：籍贯城市 <br>
fdomicile_area：籍贯县市 <br>
sch_fprovince_name：学校省份 <br>
sch_fcity_name：学校城市 <br>
sch_fregion_name：学校县市 <br>
sch_fcompany_name：学校片区名 <br>
fstd_num：在校人数 <br>
fcollege_level：学历 <br>
fcal_graduation：预计毕业时间 <br>
fauth_source_type：授信来源类型 <br>
		
### 过去六个月订单行为汇总（p6m）		
fuid_md5：MD5加密后用户ID <br>
pyear_month：观测月 <br>
cyc_date：当前观测月应还款日 <br>
od_cnt：当前观测月新建订单数 <br>
actual_od_cnt：当前观测月新建非延期分期订单数 <br>
virtual_od_cnt：当前观测月新建延期分期现金订单数 <br>
od_3c_cnt：当前观测月新建3C类订单数 <br>
od_bh_cnt：当前观测月新建百货类订单数 <br>
od_yl_cnt：当前观测月新建娱乐类订单数 <br>
od_xj_cnt：当前观测月新建现金类订单数 <br>
od_ptsh_cnt：当前观测月新建普通商户类订单数 <br>
od_zdfq_cnt：当前观测月新建账单分期类订单数 <br>
od_xssh_cnt：当前观测月新建线上商户类订单数 <br>
od_zdyq_cnt：当前观测月新建账单延期类订单数 <br>
od_lh_new_cnt：当前观测月新建新乐花类订单数 <br>
od_brw：当前观测月新建订单金额（分）(减首付后) <br>
actual_od_b：当前观测月新建延期分期现金订单金额（分）(减首付后) <br>
od_3c_brw：当前观测月新建3C类订单金额（分）(减首付后) <br>
od_bh_brw：当前观测月新建百货类订单金额（分）(减首付后) <br>
od_yl_brw：当前观测月新建娱乐类订单金额（分）(减首付后) <br>
od_xj_brw：当前观测月新建现金类订单金额（分）(减首付后) <br>
od_ptsh_brw：当前观测月新建普通商户类订单金额（分）(减首付后) <br>
od_zdfq_br：当前观测月新建账单分期类订单金额（分）(减首付后) <br>
od_xssh_brw：当前观测月新建线上商户类订单金额（分）(减首付后) <br>
od_zdyq_brw：当前观测月新建账单延期类订单金额（分）(减首付后) <br>
od_lh_new_brw：当前观测月新建新乐花类订单金额（分）(减首付后) <br>
cumu_od_cnt：历史存量创建订单数 <br>
cumu_actual_od_cnt：历史存量创建非延期分期订单数 <br>
cumu_virtual_od_cnt：历史存量创建延期分期现金订单数 <br>
cumu_od_3c_cnt：历史存量创建3C类订单数 <br>
cumu_od_bh_cnt：历史存量创建百货类订单数\<br>
cumu_od_yl_cnt：历史存量创建娱乐类订单数\<br>
cumu_od_xj_cnt：历史存量创建现金类订单数\<br>
cumu_od_ptsh_cnt：历史存量创建普通商户类订单数\<br>
cumu_od_zdfq_cnt：历史存量创建账单分期类订单数\<br>
cumu_od_xssh_cnt：历史存量创建线上商户类订单数\<br>
cumu_od_zdyq_cnt：历史存量创建账单延期类订单数\<br>
cumu_od_lh_new_cnt：历史存量创建新乐花类订单数\<br>
cumu_od_brw：历史存量创建订单金额（分）(减首付后)\<br>
cumu_actual_od_brw：历史存量创建非延期分期订单金额（分）(减首付后)\<br>
cumu_virtual_od_brw：历史存量创建延期分期现金订单金额（分）(减首付后)\<br>
cumu_od_3c_brw：历史存量创建3C类订单金额（分）(减首付后)\<br>
cumu_od_bh_brw：历史存量创建百货类订单金额（分）(减首付后)\<br>
cumu_od_yl_brw：历史存量创建娱乐类订单金额（分）(减首付后)\<br>
cumu_od_xj_brw：历史存量创建现金类订单金额（分）(减首付后)\<br>
cumu_od_ptsh_brw：历史存量创建普通商户类订单金额（分）(减首付后)\<br>
cumu_od_zdfq_brw：历史存量创建账单分期类订单金额（分）(减首付后)\<br>
cumu_od_xssh_brw：历史存量创建线上商户类订单金额（分）(减首付后)\<br>
cumu_od_zdyq_brw：历史存量创建账单延期类订单金额（分）(减首付后)\<br>
cumu_od_lh_new_brw：历史存量创建新乐花类订单金额（分）(减首付后)\<br>
payed_capital：截止到当前应还款日的已还本金（分）\<br>
payed_actual_capital：截止到当前应还款日的已还实际现金本金（分）\<br>
payed_virtual_capital：截止到当前应还款日的已还虚拟现金本金（分）\<br>
payed_3c_capital：截止到当前应还款日的已还3C类订单本金（分）\<br>
payed_bh_capital：截止到当前应还款日的已还百货类订单本金（分）\<br>
payed_yl_capital：截止到当前应还款日的已还娱乐类订单本金（分）\<br>
payed_xj_capital：截止到当前应还款日的已还现金类订单本金（分）\<br>
payed_ptsh_capital：截止到当前应还款日的已还普通商户类订单本金（分）\<br>
payed_zdfq_capital：截止到当前应还款日的已还账单分期类订单本金（分）\<br>
payed_xssh_capital：截止到当前应还款日的已还线上商户类订单本金（分）\<br>
payed_zdyq_capital：截止到当前应还款日的已还账单延期类订单本金（分）\<br>
payed_lh_new_capital：截止到当前应还款日的已还新乐花类订单本金（分）\<br>
payed_mon_fee：截止到当前应还款日的已还月服务费（分）\<br>
payed_3c_mon_fee：截止到当前应还款日的已还3C类月服务费（分）\<br>
payed_bh_mon_fee：截止到当前应还款日的已还百货类月服务费（分）\<br>
payed_yl_mon_fee：截止到当前应还款日的已还娱乐类月服务费（分）\<br>
payed_xj_mon_fee：截止到当前应还款日的已还现金类月服务费（分）\<br>
payed_ptsh_mon_fee：截止到当前应还款日的已还普通商户类月服务费（分）\<br>
payed_zdfq_mon_fee：截止到当前应还款日的已还账单分期类月服务费（分）\<br>
payed_xssh_mon_fee：截止到当前应还款日的已还线上商户类月服务费（分）\<br>
payed_zdyq_mon_fee：截止到当前应还款日的已还账单延期类月服务费（分）\<br>
payed_lh_new_mon_fee：截止到当前应还款日的已还新乐花类月服务费（分）\<br>
payed_tot_fee：截止到当前应还款日的已还总服务费（分）\<br>
payed_3c_tot_fee：截止到当前应还款日的已还3C类总服务费（分）\<br>
payed_bh_tot_fee：截止到当前应还款日的已还百货类总服务费（分）\<br>
payed_yl_tot_fee：截止到当前应还款日的已还娱乐类总服务费（分）\<br>
payed_xj_tot_fee：截止到当前应还款日的已还现金类总服务费（分）\<br>
payed_ptsh_tot_fee：截止到当前应还款日的已还普通商户类总服务费（分）\<br>
payed_zdfq_tot_fee：截止到当前应还款日的已还账单分期类总服务费（分）\<br>
payed_xssh_tot_fee：截止到当前应还款日的已还线上商户类总服务费（分）\<br>
payed_zdyq_tot_fee：截止到当前应还款日的已还账单延期类总服务费（分）\<br>
payed_lh_new_tot_fee：截止到当前应还款日的已还新乐花类总服务费（分）\<br>
bal：截止到当前应还款日的待还本金（分）\<br>
ds3c_bal：截止到当前应还款日的待还3C类订单本金（分）\<br>
bh_bal：截止到当前应还款日的待还百货类订单本金（分）\<br>
yl_bal:截止到当前应还款日的待还娱乐类订单本金（分）\<br>
xj_bal:截止到当前应还款日的待还现金类订单本金（分）\<br>
ptsh_bal:截止到当前应还款日的待还普通商户类订单本金（分）\<br>
zdfq_bal:截止到当前应还款日的待还账单分期类订单本金（分）\<br>
xssh_bal:截止到当前应还款日的待还线上商户类订单本金（分）\<br>
zdyq_bal:截止到当前应还款日的待还账单延期类订单本金（分）\<br>
lh_new_bal:截止到当前应还款日的待还新乐花类订单本金（分）\<br>
paying_mon_fee:截止到当前应还款日的应还服务费（分）\<br>
ds3c_paying_mon_fee:截止到当前应还款日的应还3C类月服务费（分）\<br>
bh_paying_mon_fee:截止到当前应还款日的应还百货类月服务费（分）\<br>
yl_paying_mon_fee:截止到当前应还款日的应还娱乐类月服务费（分）\<br>
xj_paying_mon_fee:截止到当前应还款日的应还现金类月服务费（分）\<br>
ptsh_paying_mon_fee:截止到当前应还款日的应还普通商户类月服务费（分）\<br>
zdfq_paying_mon_fee:截止到当前应还款日的应还账单分期类月服务费（分）\<br>
xssh_paying_mon_fee:截止到当前应还款日的应还线上商户类月服务费（分）\<br>
zdyq_paying_mon_fee:截止到当前应还款日的应还账单延期类月服务费（分）\<br>
lh_new_paying_mon_fee:截止到当前应还款日的应还新乐花类月服务费（分）\<br>
paying_tot_fee:截止到当前应还款日的应还本金（分）\<br>
ds3c_paying_tot_fee:截止到当前应还款日的应还3C类总服务费（分）\<br>
bh_paying_tot_fee:截止到当前应还款日的应还百货类总服务费（分）\<br>
yl_paying_tot_fee:截止到当前应还款日的应还娱乐类总服务费（分）\<br>
xj_paying_tot_fee:截止到当前应还款日的应还现金类总服务费（分）\<br>
ptsh_paying_tot_fee:截止到当前应还款日的应还普通商户类总服务费（分）\<br>
zdfq_paying_tot_fee:截止到当前应还款日的应还账单分期类总服务费（分）\<br>
xssh_paying_tot_fee:截止到当前应还款日的应还线上商户类总服务费（分）\<br>
zdyq_paying_tot_fee:截止到当前应还款日的应还账单延期类总服务费（分）\<br>
lh_new_paying_tot_fee:截止到当前应还款日的应还新乐花类总服务费（分）\<br>
paying_complete_od_cnt：截止到当前应还款日的应全部还完订单数\<br>
payed_complete_od_cnt：截止到当前应还款日的已全部还完订单数\<br>
payed_complete_actual_od_cnt：截止到当前应还款日的已全部实际现金还款还完订单数\<br>
paying_complete_od_brw：截止到当前应还款日的应全部还完订单金额（分）\<br>
payed_complete_od_brw：截止到当前应还款日的已全部还完订单金额（分）\<br>
payed_complete_actual_od_brw：截止到当前应还款日的已全部实际现金还款还完订单金额（分）\<br>
acre_repay_od_cnt：截止到当前应还款日的历史提前还款订单\<br>
acre_repay_od_cpt：截止到当前应还款日的历史提前还款的本金金额（分）\<br>
foverdue_paying_day：用户当前逾期天数\<br>
foverdue_paying_cyc：用户当前逾期账期数\<br>
foverdue_payed_day：用户历史逾期天数\<br>
foverdue_payed_cyc：用户历史逾期账期数\<br>
cpt_pymt：当前观测月还款额（分）\<br>
credit_limit：当前观测月额度（分）\<br>
fcredit_update_time：额度更新时间\<br>
futilization：额度使用率\<br>
fopen_to_buy：剩余可用额度（分）\<br>
\<br>
###未来六个月的消费信息及dep\<br>	
fuid_md5：MD5加密后用户ID\<br>
dep：未来6个月是否有过逾期3个账期以上\<br>
actual_od_brw_f6m：未来6个月新建非延期分期总订单金额（分）\<br>
actual_od_brw_1stm：未来第一月新建非延期分期订单金额（分）\<br>
actual_od_brw_2stm：未来第二月新建非延期分期订单金额（分）\<br>
actual_od_brw_3stm：未来第三月新建非延期分期订单金额（分）\<br>
actual_od_brw_4stm：未来第四月新建非延期分期订单金额（分）\<br>
actual_od_brw_5stm：未来第五月新建非延期分期订单金额（分）\<br>
actual_od_brw_6stm：未来第六月新建非延期分期订单金额（分）\<br>
