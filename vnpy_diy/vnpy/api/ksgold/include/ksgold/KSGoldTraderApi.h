#ifndef __KSGOLDTRADEAPI__H__
#define __KSGOLDTRADEAPI__H__

#ifdef KSGOLDTRADERAPI_EXPORTS
#define KSGOLDTRADERAPI_API __declspec(dllexport)
#else
#ifdef WIN32
#define KSGOLDTRADERAPI_API __declspec(dllimport)
#else 
#define KSGOLDTRADERAPI_API
#endif
#endif

#include "KSGoldUserApiStructEx.h"

namespace KSGoldTradeAPI
{

class CKSGoldTraderSpi
{
public:
	///���ͻ����뽻�׺�̨������ͨ������ʱ���÷��������á�
	///���ֶ�������ʱ��Ҳ����ô˷���
	///@param nResult ���ؽ��
	///        0x1001 ��������
	///        0x1002 ���������ɹ�
	virtual void OnFrontConnected(int nResult) {};
	
	///���ͻ����뽻�׺�̨ͨ�����ӶϿ�ʱ���÷��������á���������������API���Զ��������ӣ��ͻ��˿ɲ�������
	///@param nReason ����ԭ��
	///        0x1001 �����ʧ��
	///        0x1002 ����дʧ��
	///        0x2001 ����������ʱ
	///        0x2002 ��������ʧ��
	///        0x2003 �յ�������
	virtual void OnFrontDisconnected(int nReason) {};
	
	///��¼������Ӧ
	virtual void OnRspUserLogin(CThostFtdcRspUserLoginField *pRspUserLogin, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	///�ǳ�������Ӧ
	virtual void OnRspUserLogout(CThostFtdcUserLogoutField *pUserLogout, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	//�г�״̬֪ͨ
	virtual void OnNtyMktStatus(CThostFtdcMarketStatusField *pfldMktStatus){};
	
	///��Լ����״̬֪ͨ
	virtual void OnRtnInstrumentStatus(CThostFtdcInstrumentStatusField *pInstrumentStatus){};
	
	///�����ѯ��Լ��Ӧ
	virtual void OnRspQryInstrument(CThostFtdcInstrumentField *pInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	//�����ѯ����Ʒ����Ӧ
	virtual void OnRspReqQryVarietyCode(CThostFtdcRspVarietyCodeField *pVarietyCode, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};

	///����¼��������Ӧ
	virtual void OnRspOrderInsert(CThostFtdcRspInputOrderField *pRspInputOrder, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	///ETF�Ϲ�������Ӧ
	virtual void OnRspETFSubscriptionOrderInsert(CThostFtdcETFTradeDetailField *pETFSubscriptionOrderInsert, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///ETF�깺������Ӧ
	virtual void OnRspETFPurchaseOrderInsert(CThostFtdcETFTradeDetailField *pETFSubscriptionOrderInsert, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///ETF���������Ӧ
	virtual void OnRspETFRedeemInsert(CThostFtdcETFTradeDetailField *pETFSubscriptionOrderInsert, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///ETF�˻���������Ӧ
	virtual void OnRspETFAccountBinding(CThostFtdcETFBindingStatusField *pETFAccountBinding, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///ETF�˻����������Ӧ
	virtual void OnRspETFAccountUnbinding(CThostFtdcETFBindingStatusField *pETFAccountUnbinding, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///����֪ͨ
	virtual void OnRtnOrder(CThostFtdcOrderField *pOrder) {};
	
	///ǿ��֪ͨ
	virtual void OnForceLogout(CThostFtdcUserLogoutField *pLogout) {};

	//ETF�˻��󶨽��״̬֪ͨ
	virtual void OnRtnETFAccountBindingStatus(CThostFtdcETFBindingStatusField * pETFBindgingStatus) {};

	//ETF�����걨��״̬֪ͨ
	virtual void OnRtnETFOrder(CThostFtdcETFTradeDetailField *pEtfTradeDetail){};
	
	///����Ӧ��
	virtual void OnRspOrderAction(CThostFtdcRspInputOrderActionField *pRspInputOrderAction, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};

	///����Ӧ��
	virtual void OnRspError(CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};

	///�ɽ�֪ͨ
	virtual void OnRtnTrade(CThostFtdcTradeField *pTrade) {};
	
	///�����ѯ�ʽ��˻���Ӧ
	virtual void OnRspQryTradingAccount(CThostFtdcTradingAccountField *pTradingAccount, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	///������ʷ�ʽ��ѯ
	virtual void OnRspQryHisCapital(CThostFtdcRspHisCapitalField *pHisCapital, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};

	///�����ѯ������Ӧ
	virtual void OnRspQryOrder(CThostFtdcOrderField *pOrder, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	///�����ѯ�ɽ���Ӧ
	virtual void OnRspQryTrade(CThostFtdcTradeField *pTrade, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	///�����ѯͶ���ֲ߳���Ӧ
	virtual void OnRspQryInvestorPosition(CThostFtdcInvestorPositionField *pInvestorPosition, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};
	
	///��ѯ�ͻ������Ӧ
	virtual void OnRspQryClientStorage(CThostFtdcStorageField *pStorage, CThostFtdcRspInfoField *pRspInfo,int nRequestID,bool bIsLast){};
	
	///����\��֤���ʲ�ѯ��Ӧ
	virtual void OnRspQryCostMarginFeeRate(CThostFtdcRspCostMarginFeeField *pCostMarginFee, CThostFtdcRspInfoField *pRspInfo,int nRequestID, bool bIsLast){};
	
	///������ί����Ӧ
	virtual void OnRspConditionOrderInsert(CThostFtdcRspConditionOrderField *pConditionOrder, CThostFtdcRspInfoField *pRspInfo,int nRequestID, bool bIsLast){};
	
	///������������Ӧ
	virtual void OnRspConditionOrderAction(CThostFtdcRspConditionActionOrderField *pConditionActionOrder, CThostFtdcRspInfoField *pRspInfo,int nRequestID, bool bIsLast){};
	
	///������ί�в�ѯ��Ӧ
	virtual void OnRspQryConditionOrder(CThostFtdcRspConditionOrderQryField *pConditionOrderQry, CThostFtdcRspInfoField *pRspInfo,int nRequestID, bool bIsLast){};
	
	///�������ɽ���ѯ��Ӧ
	virtual void OnRspQryConditionOrderTrade(CThostFtdcRspConditionOrderMatchField *pConditionOrderMatch, CThostFtdcRspInfoField *pRspInfo,int nRequestID, bool bIsLast){};

	///�ͻ��Ự��Ϣͨ�ò�ѯ��Ӧ
	virtual void OnRspQryClientSessionInfo(CThostFtdcRspClientSessionField *pClientSessionField, CThostFtdcRspInfoField *pRspInfo,int nRequestID, bool bIsLast){};

	///��ѯ������Ϣ��Ӧ
	virtual void OnRspQryQuotation(CThostFtdcDepthMarketDataField *pDepthMarketData, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};	

	///��ѯͶ���ֲ߳���ϸ��Ӧ
	virtual void OnRspQryInvestorPositionDetail(CThostFtdcInvestorPositionDetailField *pInvestorPositionDetail, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///ETF�����꽻�ײ�ѯ��Ӧ
	virtual void OnRspQryETFradeDetail(CThostFtdcETFTradeDetailField *pQryETFTradeDetail, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///ETF�������嵥��ѯ
	virtual void OnRspQryETFPcfDetail(CThostFtdcETFPcfDetailField *pQryETFpcfDetail, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};

	///�޸�������Ӧ
	virtual void OnRspModifyPassword(CThostFtdcModifyPasswordRsqField *pRsqModifyPassword,CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};
	
	///���г������Ӧ
	virtual void OnRspB0CMoneyIO(CThostFtdcBOCMoneyIORspField *pRspBOCMoneyIO,CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast){};


};

class KSGOLDTRADERAPI_API CKSGoldTraderApi
{
public:
	///����TraderApi
	///@param pszFlowPath �������ļ�Ŀ¼��Ĭ��Ϊ��ǰĿ¼
	///@return ��������UserApi
	static CKSGoldTraderApi *CreateGoldTraderApi(const char *pszFlowPath = "");
	
	///ɾ���ӿڶ�����
	///@remark ����ʹ�ñ��ӿڶ���ʱ,���øú���ɾ���ӿڶ���
	virtual void Release() = 0;
	
	///��ʼ��
	///@remark ��ʼ�����л���,ֻ�е��ú�,�ӿڲſ�ʼ����
	virtual bool Init() = 0;
	
	///ע��ǰ�û������ַ
	///@param pszFrontAddress��ǰ�û������ַ��
	///@remark �����ַ�ĸ�ʽΪ����protocol://ipaddress:port�����磺��tcp://127.0.0.1:17001���� 
	///@remark ��tcp��������Э�飬��127.0.0.1�������������ַ����17001������������˿ںš�
	virtual void RegisterFront(char *pszFrontAddress) = 0;
	
	///����˽������
	///@param nResumeType ˽�����ش���ʽ  
	///        KS_TERT_RESTART: �ӱ������տ�ʼ�ش�
	///        KS_TERT_RESUME : ���ϴ��յ�������,������ʱ������
	///        KS_TERT_QUICK  : ֻ���͵�¼��˽����������
	///@remark �÷���Ҫ��Init����ǰ���á�����������Ĭ����KS_TERT_RESTART��
	virtual void SubscribePrivateTopic(KS_TE_RESUME_TYPE nResumeType) = 0;

	///���Ĺ�������
	///@param nResumeType �������ش���ʽ  
	///        KS_TERT_RESTART: �ӱ������տ�ʼ�ش�
	///        KS_TERT_RESUME : ���ϴ��յ�������,������ʱ������
	///        KS_TERT_QUICK  : ֻ���͵�¼�󹫹���������
	///@remark �÷���Ҫ��Init����ǰ���á�����������Ĭ����KS_TERT_RESTART��
	virtual void SubscribePublicTopic(KS_TE_RESUME_TYPE nResumeType) = 0;

	///ע��API�ص��ӿ�
	///@param pGeneralSpi �����Իص��ӿ����ʵ��
	virtual void RegisterSpi(CKSGoldTraderSpi *pGeneralSpi) = 0;	
	
	///�ȴ��ӿ��߳̽�������
	///@return �߳��˳�����
	virtual int Join() = 0;
	
	///�û���¼����
	virtual int ReqUserLogin(CThostFtdcReqUserLoginField *pReqUserLoginField, int nRequestID) = 0;
	
	///�û��ǳ�����
	virtual int ReqUserLogout(CThostFtdcUserLogoutField *pUserLogout, int nRequestID) = 0;
	
	///��ѯ��Լ
	virtual int ReqQryInstrument(CThostFtdcQryInstrumentField *pQryInstrument, int nRequestID) = 0;

	///��ѯ����Ʒ��
	virtual int ReqQryVarietyCode(CThostFtdcQryVarietyCodeField *pQryVarietyCode, int nRequestID) = 0;
	
	///�µ�����
	virtual int ReqOrderInsert(CThostFtdcInputOrderField *pInputOrder, int nRequestID) = 0;
	
	///������������(����)
	virtual int ReqOrderAction(CThostFtdcInputOrderActionField *pInputOrderAction, int nRequestID) = 0;
	
	///��ѯ�ֲ�����
	virtual int ReqQryInvestorPosition(CThostFtdcQryInvestorPositionField *pInvestorPosition, int nRequestID) = 0;
	
	///��ѯ�ʽ�����
	virtual int ReqQryTradingAccount(CThostFtdcQryTradingAccountField *pQryTradingAccount, int nRequestID) = 0;
	
	///�ɽ���ѯ����
	virtual int ReqQryTrade(CThostFtdcQryTradeField *pQryTrade, int nRequestID) = 0;
	
	///ί�в�ѯ����
	virtual int ReqQryOrder(CThostFtdcQryOrderField *pQryOrder, int nRequestID) = 0;
	
	///��ѯ�ͻ��������
	virtual int ReqQryStorage(CThostFtdcQryStorageField *pfldStorage, int nRequestID) =0;
	
	//����\��֤���ʲ�ѯ
	virtual int ReqQryCostMarginFeeRate(CThostFtdcQryCostMarginFeeField *pQryCostMarginFee, int nRequestID) = 0;
	
	///������ί��
	virtual int ReqConditionOrderInsert(CThostFtdcConditionOrderField *pConditionOrder, int nRequestID) = 0;
	
	///����������
	virtual int ReqConditionOrderAction(CThostFtdcConditionActionOrderField *pConditionActionOrder, int nRequestID) = 0;
	
	///������ί�в�ѯ
	virtual int ReqQryConditionOrder(CThostFtdcConditionOrderQryField *pConditionOrderQry, int nRequestID) = 0;
	
	///�������ɽ���ѯ
	virtual int ReqQryConditionOrderTrade(CThostFtdcConditionOrderMatchField *pConditionOrderMatch, int nRequestID) = 0;

	///�ͻ��Ự��Ϣͨ�ò�ѯ
	virtual int ReqQryClientSessionInfo(CThostFtdcQryClientSessionField *pQryClientSession, int nRequestID) = 0;
	
	///��ѯ������Ϣ
	virtual int ReqQryQuotation(CThostFtdcQryQuotationField *pQryQuotation, int nRequestID) = 0;

	///�����ѯͶ���ֲ߳���ϸ
	virtual int ReqQryInvestorPositionDetail(CThostFtdcQryInvestorPositionDetailField *pQryInvestorPositionDetail, int nRequestID) = 0;

	///�޸���������
	virtual int ReqModifyPassword(CThostFtdcModifyPasswordField *pModifyPasswordFieldl, int nRequestID) = 0;

    ///�����ѯ�ͻ���ʷ�ʽ�
	virtual int ReqQryHisCapital(CThostFtdcQryHisCapitalField *pQryHisCapital,int nRequestID) = 0;

	///ETF�Ϲ�����
	virtual int ReqETFSubScription(CThostFtdcSubScriptionField *pEtfSubScription,int nRequestID) = 0;

    ///ETF�깺����
	virtual int ReqETFApplyForPurchase(CThostFtdcApplyForPurchaseField *pEtfApplyForPurchase,int nRequestID) = 0;

	///ETF�������
	virtual int ReqETFRedeem(CThostFtdcRedeemField *pETFRedeem,int nRequestID)=0;

	///ETF�˻���
	virtual int ReqETFAccountBinding(CThostFtdcETFBingingField *pETFAccountBinding,int nRequestID)=0;

	///ETF�˻����
	virtual int ReqETFAccountUnbinding(CThostFtdcETFUnBingingField *pETFAccountUnBinding,int nRequestID)=0;

	///ETF�����꽻�ײ�ѯ
	virtual int ReqETFTradeDetail(CThostFtdcQryETFTradeDetailField *pQryETFTradeDetail,int nRequestID) = 0;
 
	///ETF�������嵥��ѯ
	virtual int ReqETFPcfDetail(CThostFtdcQryETFPcfDetailField *pQryETFPcfDetail,int nRequestID) = 0;

	///���г��������
	virtual int ReqBOCMoneyIO(CThostFtdcBOCMoneyIOField *pBOCMoneyIO, int nRequestID) = 0;


	protected:
		~CKSGoldTraderApi();
};

}  //end of namespace KSGoldTradeAPI

#endif   //__KSGOLDTRADEAPI__H__




