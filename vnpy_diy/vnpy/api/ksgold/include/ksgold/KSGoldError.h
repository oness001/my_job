#ifndef __KSGOLDERROR__H__
#define __KSGOLDERROR__H__

#define KSGOLDERROR_INPUTNULL							1000    //���ΪNULL
#define KSGOLDERROR_NOTLOGIN							1001    //�ͻ���δ��¼
#define KSGOLDERROR_PARAMETER_EXCEPTION					1002    //���ͱ��ĵĲ����쳣
#define KSGOLDERROR_OVERLOAD							1003    //���󳬹�δ�����������
#define KSGOLDERROR_NOTCONNECTGATEWAY					1004    //APIδ��������
#define KSGOLDERROR_ALREADLOGIN							1005    //�ÿͻ�ʵ���Ѿ���¼
#define KSGOLDERROR_PRICEILLEGAL						1006    //����ļ۸�Ƿ�
#define KSGOLDERROR_ORDERREFTOOLOW						1007    //�ƽ���ƽ̨�µ�ʧ��[�������������С]
#define KSGOLDERROR_GATAWAYINITFAIL						1008    //���س�ʼ��ʧ��
#define KSGOLDERROR_ORDERREFSESSIONIDCHECKFAIL			1009    //������źͻỰID��ƥ��
#define KSGOLDERROR_FLOWCONTROLCHECKFAIL				1010	//API��������
#define KSGOLDERROR_FIRSTLOGINNEEDMODPWD				1099	//�״ε�¼��Ҫ���޸�����
#define KSGOLDERROR_VERSIONCHECKFAIL					-1011   //�汾У��ʧ��
#define KSGOLDERROR_TRADERACCOUNTFUND					-1012   //�ͻ��˻��ʽ��ѯʧ��
#define KSGOLDERROR_USERLOGINEFAIL						-1013   //�ͻ���¼ʧ��...
#define KSGOLDERROR_USERLOGOUTFAIL						-1014   //�ͻ�¼��ʧ��...
#define KSGOLDERROR_SPOTINSTQUERYFAIL					-1015	//�ֻ���Լ��ѯʧ��
#define KSGOLDERROR_TNINSTQUERYFAIL						-1016	//����T+N��Լ��ѯʧ��
#define KSGOLDERROR_DEFERINSTQUERYFAIL					-1017	//���Ӻ�Լ��ѯʧ��
#define KSGOLDERROR_DELIVERYCODEQUERYFAIL				-1018	//����Ʒ�ֲ�ѯʧ��
#define KSGOLDERROR_INSTCACHNOTEXIST					-1019	//����ĺ�Լ��Ϣ������
#define KSGOLDERROR_INSTNOTFOUND						-1020	//δ��ѯ����Ӧ�ĺ�Լ��Ϣ
#define KSGOLDERROR_DELIVERYCODECACHNOTEXIST			-1021	//����Ľ���Ʒ����Ϣ������
#define KSGOLDERROR_DELIVERYCODENOTFOUND				-1022	//δ��ѯ����Ӧ�Ľ���Ʒ����Ϣ
#define KSGOLDERROR_INITORDERQUERYERROR					-1023	//��ʼ��ί�в�ѯʧ��
#define KSGOLDERROR_INITDONEQUERYERROR					-1024	//��ʼ���ɽ���ѯʧ��
#define KSGOLDERROR_INITDONEQUERYGETNEXTERROR			-1025	//��ʼ��ί�в�ѯȡ����������
#define KSGOLDERROR_INITORDERQUERYGETNEXTERROR			-1026	//��ʼ���ɽ���ѯȡ���������� 
#define KSGOLDERROR_QUERYCONNECTIONLOGINFAIL			-1027	//��ѯ���ӵ�¼ʧ��...
#define KSGOLDERROR_ORDERACTIONFAIL						-1028	//��ͨ������������������...
#define KSGOLDERROR_CONDITIONORDERACTIONFAIL			-1029   //������������������������,�������Ų���Ϊ��
#define KSGOLDERROR_SUBMARKETDATAOPERATORFAIL			-1030   //���鶩��/�˶�����ʧ��
#define KSGOLDERROR_QUERYRECORDNOTFOUND					-1031   //���޼�¼

#endif  //__KSGOLDERROR__H__