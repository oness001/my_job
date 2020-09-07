#ifndef __KSGOLDTRADEDEFINE_H__
#define __KSGOLDTRADEDEFINE_H__
//�г����� ������:TMarketType
#define KS_SPOT			"00"     //�ֻ�
#define KS_DEFER		"10"     //����
#define KS_FUTURES		"11"     //�ڻ�
#define KS_FORWARD		"01"     //Զ��
#define KS_DELIVERY		"AP"	 //����
#define KS_MIDDLE		"MD"	 //������

//��Լ״̬
#define KS_I_INITING		'0'     //��ʼ����
#define KS_I_INIT			'1'     //��ʼ�����
#define KS_I_BEGIN			'2'     //����
#define KS_I_GRP_ORDER		'3'     //���۱���
#define KS_I_GRP_MATCH		'4'     //���۴��
#define KS_I_NORMAL			'5'     //��������
#define KS_I_PAUSE			'6'     //��ͣ
#define KS_I_DERY_APP		'7'     //�����걨
#define KS_I_DERY_MATCH		'8'     //�����걨����
#define KS_I_MID_APP		'9'     //�������걨
#define KS_I_MID_MATCH		'A'     //�����걨���
#define KS_I_END			'B'     //����

//�г�״̬
#define KS_M_INITING	'0'     //��ʼ����
#define KS_M_INIT		'1'     //��ʼ�����
#define KS_M_OPEN		'2'     //����
#define KS_M_TRADE		'3'     //����
#define KS_M_PAUSE		'4'     //��ͣ
#define KS_M_CLOSE		'5'     //����
 
//��ͨί�е�״̬
#define KS_Entrust_Sending				'1'			//�Ѿ����ͣ�
#define KS_Entrust_Waiting				'2'			//�ȴ�����
#define KS_Entrust_In					'3'			//�Ѿ����룻
#define KS_Entrust_All_Cancel			'4'			//ȫ��������
#define KS_Entrust_All_Done				'5'			//ȫ���ɽ���
#define KS_Entrust_Part_Done			'6'			//���ֳɽ���
#define KS_Entrust_Part_Done_Cancel		'7'			//���ɲ���
#define KS_Entrust_Wait_Cancel			'8'			//�ȴ�����
#define KS_Entrust_Error				'9'			//����ί��
#define KS_Entrust_By_Exch				'A'			//���ڳ���
#define KS_Entrust_By_Emergency			'B'			//Ӧ������
#define KS_Entrust_By_Auto_Cancel		'C'			//�Զ�����

//������ί�м۸�����
#define  KS_CONDITION_MARKET_PRICE       '0'		//�м�
#define  KS_CONDITION_BUYFIVE_PRICE		 '1'		//���嵵
#define  KS_CONDITION_SELLFIVE_PRICE	 '2'		//���嵵
#define  KS_CONDITION_COMMON_PRICE		 '3'        //��ͨί��(����)
#define  KS_CONDITION_BUYONE_PRICE		 '4'		//��һ��
#define  KS_CONDITION_SELLONE_PRICE		 '5'		//��һ��
#define  KS_CONDITION_BUYTWO_PRICE		 '6'		//�����
#define  KS_CONDITION_SELLTWO_PRICE		 '7'		//������
#define  KS_CONDITION_BUYTHREE_PRICE	 '8'		//������
#define  KS_CONDITION_SELLTHREE_PRICE	 '9'		//������
#define  KS_CONDITION_BUYFOUR_PRICE		 'a'		//���ļ�
#define  KS_CONDITION_SELLFOUR_PRICE	 'b'		//���ļ� 

//������ί������
#define	 KS_CONDITION_PROFIT_ORDER       '0'		//ֹӯ��,���ڵ��ڴ�����
#define	 KS_CONDITION_LOSS_ORDER         '1'		//ֹ��,С�ڵ��ڴ�����

//������ί��״̬
#define  KS_CONDITION_ORDER_NOTTRIGGER	 '0'		//δ����
#define  KS_CONDITION_ORDER_SUCCESS		 '1'		//�Ѵ���-����ɹ�
#define  KS_CONDITION_ORDER_STATUS		 '2'		//�Ѵ���-����ʧ��
#define  KS_CONDITION_ORDER_NOTUSE		 '3'		//������-�ͻ�����
#define  KS_CONDITION_ORDER_EXPIRED		 '4'		//������-ϵͳ��������
#define  KS_CONDITION_ORDER_PROCESSING	 '5'		//���ڴ���

//��ƽ�ֱ�־ ������:TOffsetFlag
#define KS_P_OPEN		    '0'     //����
#define KS_P_OFFSET			'1'     //ƽ��


//���������־ ������:TBSFlag
#define KS_BUY		'0'     //��
#define KS_SELL		'1'     //��

//ί������
#define  KS_SPOT_ENTRUST     '0' //�ֻ�
#define  KS_TN_ENTRUST       '4' //�ֻ�T+N
#define  KS_DEFER_ENTRUST    '1' //�ֻ�����
#define  KS_DELIVERY_ENTRUST '2' //����
#define  KS_MIDDLE_ENTRUST   '3' //������

//��������
#define  KS_COUNTER_CHANNEL  '1' //��̨
#define  KS_TEL_CHANNEL      '2' //�绰����
#define  KS_NET_CHANNEL      '3' //����
#define  KS_TRADER_CHANNEL   '4' //����Ա
#define  KS_SELF_CHANNEL	 '5' //�����ն�
#define  KS_PHONE_CHANNEL	 '6' //�ֻ�����
#define  KS_TRADEAPI_CHANNEL '7' //����API

//�ͻ����
#define KS_CLIENT_SPOT    '0' //�ֻ�
#define KS_CLIENT_FUTURE  '1' //�ڻ�
#define KS_CLIENT_GENERAL '2' //ͨ��

//��¼����
#define KS_BANKACC_TYPE   "1" //�����˺ŵ�¼
#define KS_TRADECODE_TYPE "2" //���ױ����¼

//��Ծ��־
#define KS_ACTIVITY_ON '1' //ѯ�ۺ�Լ

//Ʒ�����
#define KS_VARIETY_CODE_GOLD   0    //�ƽ�
#define KS_VARIETY_CODE_AG     1    //����
#define KS_VARIETY_CODE_PT     2    //����

#define KS_LOG_CLEAR_DAY 3 // ������־�������

// ��־��ʼ��
#define KS_BEGIN_	  "$" 
// ��־�ָ���
#define KS_SPLIT_	  "|" 

//API����״̬��ʶ
#define KS_FRONTCONNECTED		1		//������
#define KS_DISCONNECTED			2       //�Ͽ�����
#define KS_RECONNECTING			3       //��������
#define KS_ONSTATUSCHANGE		4       //״̬�ı�

//API״̬����ʶ
#define KS_STATE_INIT			'0'		//��ʼ״̬
#define KS_STATE_READY			'1'		//����״̬
#define KS_STATE_ONLINE			'2'		//����״̬
#define	KS_STATE_LOGON			'3'		//¼��״̬

//ָ������
#define KS_COMM_INSTRUCTION		   '0'     //��ָͨ��
#define KS_FOK_INSTRUCTION		   '1'     //FOKָ��
#define KS_FAK_INSTRUCTION		   '2'     //FAKָ��
#define KS_MP2FP_INSTUCTION        '3'	   //�м�ʣ��ת�޼�
#define KS_MPFOK_INSTUCTION        '4'     //�м�FOKָ��
#define KS_MPFAK_INSTUCTION        '5'     //�м�FAKָ��	

//ETF��������
#define ETF_SCRIPTION           "020"   //�Ϲ�
#define ETF_PURCHASE            "022"   //�깺
#define ETF_REDEEM              "024"   //���

//ETF����״̬
#define ETF_PURCHASE_REQ					"101"   //�깺����
#define ETF_PURCHASE_RETURN_SUCCESS         "102"   //�깺�ر��ɹ�
#define ETF_PURCHASE_RETURN_FAIL            "103"   //�깺�ر�ʧ��
#define ETF_PURCHASE_CONFIRM_SUCCESS        "104"   //�깺ȷ�ϳɹ�
#define ETF_PURCHASE_CONFIRM_FAIL           "105"   //�깺ȷ��ʧ��
#define ETF_SCRIPTION_REQ					"201"   //�Ϲ�����
#define ETF_SCRIPTION_RETURN_SUCCESS        "202"   //�Ϲ��ر��ɹ�
#define ETF_SCRIPTION_RETURN_FAIL           "203"   //�Ϲ��ر�ʧ��
#define ETF_SCRIPTION_SUCCESS				"204"   //�Ϲ��ɹ�
#define ETF_SCRIPTION_FAIL					"205"   //�Ϲ�ʧ��
#define ETF_SCRIPTION_CONFIRM_SUCCESS	    "206"   //�Ϲ�ȷ�ϳɹ�
#define ETF_SCRIPTION_CONFIRM_FAIL          "207"   //�Ϲ�ȷ��ʧ��
#define ETF_TRANSFER_REQ			        "208"   //��������
#define ETF_REDEEM_REQ						"301"   //�������
#define ETF_REDEEM_CONFIRM_SUCCESS          "304"   //���ȷ�ϳɹ�
#define ETF_REDEEM_TRANSFER_FAIL			"305"   //��ع���ʧ��
#define ETF_FROZEN_REQ						"401"   //��������
#define ETF_FROZEN_SUCCESS					"402"   //����ɹ�
#define ETF_FROZEN_FAIL						"403"   //����ʧ��
#define ETF_TRANSFER_SUCCESS				"404"   //�����ɹ�S
#define ETF_TRANSFER_FAIL					"405"   //����ʧ��
#define ETF_UNFROZEN_SUCCESS				"406"   //�ⶳ�ɹ�
#define ETF_UNFROZEN_FAIL					"407"   //�ⶳʧ��
#define ETF_FUNDUNIT_DECREASE_REQ           "501"   //�ݶ���������
#define ETF_FUNDUNIT_DECREASE_SUCCESS       "502"   //�ݶ����ӳɹ�
#define ETF_FUNDUNIT_DECREASE_FAIL          "503"   //�ݶ�����ʧ��
#define ETF_FUNDUNIT_INCREASE_REQ           "601"   //�ݶ��������
#define ETF_FUNDUNIT_INCREASE_SUCCESS		"602"   //�ݶ���ٳɹ�
#define ETF_FUNDUNIT_INCREASE_FAIL          "603"   //�ݶ����ʧ��
#define ETF_ERR_ENTRUST                     "901"   //����ί��


#define ETF_ACCOUNT_UNBINDING                               "000"   //δ��
#define ETF_ACCOUNT_BINDING_WAITIN_GCONFIRM                 "701"   //�󶨴�ȷ��
#define ETF_ACCOUNT_BINDING_FINISH                          "702"   //�������
#define ETF_ACCOUNT_BINDING_FAIL                            "703"   //����ʧ��
#define ETF_ACCOUNT_BINDING_SEND                            "704"   //���ѷ���
#define ETF_ACCOUNT_UNBINDING_WAITIN_GCONFIRM               "801"   //��󶨴�ȷ��
#define ETF_ACCOUNT_UNBINDING_FINISH                        "802"   //��������
#define ETF_ACCOUNT_UNBINDING_FAIL                          "803"   //�����ʧ��
#define ETF_ACCOUNT_UNBINDING_SEND                          "804"   //����ѷ���

//API�����ж�����
#define KS_CONDITION_FIRSTLOGIN	1		//�״ε�¼

#endif //__KSGOLDTRADEDEFINE_H__
