//ϵͳ
#ifdef WIN32
#include "stdafx.h"
#endif

#include "vnksgold.h"
#include "pybind11/pybind11.h"
#include "ksgold/KSGoldTraderApi.h"


using namespace pybind11;
using namespace KSGoldTradeAPI;


//����
#define ONFRONTCONNECTED 0
#define ONFRONTDISCONNECTED 1
#define ONRSPUSERLOGIN 2
#define ONRSPUSERLOGOUT 3
#define ONNTYMKTSTATUS 4
#define ONRTNINSTRUMENTSTATUS 5
#define ONRSPQRYINSTRUMENT 6
#define ONRSPREQQRYVARIETYCODE 7
#define ONRSPORDERINSERT 8
#define ONRSPETFSUBSCRIPTIONORDERINSERT 9
#define ONRSPETFPURCHASEORDERINSERT 10
#define ONRSPETFREDEEMINSERT 11
#define ONRSPETFACCOUNTBINDING 12
#define ONRSPETFACCOUNTUNBINDING 13
#define ONRTNORDER 14
#define ONFORCELOGOUT 15
#define ONRTNETFACCOUNTBINDINGSTATUS 16
#define ONRTNETFORDER 17
#define ONRSPORDERACTION 18
#define ONRSPERROR 19
#define ONRTNTRADE 20
#define ONRSPQRYTRADINGACCOUNT 21
#define ONRSPQRYHISCAPITAL 22
#define ONRSPQRYORDER 23
#define ONRSPQRYTRADE 24
#define ONRSPQRYINVESTORPOSITION 25
#define ONRSPQRYCLIENTSTORAGE 26
#define ONRSPQRYCOSTMARGINFEERATE 27
#define ONRSPCONDITIONORDERINSERT 28
#define ONRSPCONDITIONORDERACTION 29
#define ONRSPQRYCONDITIONORDER 30
#define ONRSPQRYCONDITIONORDERTRADE 31
#define ONRSPQRYCLIENTSESSIONINFO 32
#define ONRSPQRYQUOTATION 33
#define ONRSPQRYINVESTORPOSITIONDETAIL 34
#define ONRSPQRYETFRADEDETAIL 35
#define ONRSPQRYETFPCFDETAIL 36
#define ONRSPMODIFYPASSWORD 37
#define ONRSPB0CMONEYIO 38



///-------------------------------------------------------------------------------------
///C++ SPI�Ļص���������ʵ��
///-------------------------------------------------------------------------------------

//API�ļ̳�ʵ��
class TdApi : public CKSGoldTraderSpi
{
private:
	CKSGoldTraderApi* api;                     //API����
    thread task_thread;                    //�����߳�ָ�루��python���������ݣ�
    TaskQueue task_queue;                //�������
    bool active = false;                //����״̬

public:
    TdApi()
    {
    };

    ~TdApi()
    {
        if (this->active)
        {
            this->exit();
        }
    };

    //-------------------------------------------------------------------------------------
    //API�ص�����
    //-------------------------------------------------------------------------------------

	///���ͻ����뽻�׺�̨������ͨ������ʱ���÷��������á�
	///���ֶ�������ʱ��Ҳ����ô˷���
	///@param nResult ���ؽ��
	///        0x1001 ��������
	///        0x1002 ���������ɹ�
	virtual void OnFrontConnected(int nResult);

	///���ͻ����뽻�׺�̨ͨ�����ӶϿ�ʱ���÷��������á���������������API���Զ��������ӣ��ͻ��˿ɲ�������
	///@param nReason ����ԭ��
	///        0x1001 �����ʧ��
	///        0x1002 ����дʧ��
	///        0x2001 ����������ʱ
	///        0x2002 ��������ʧ��
	///        0x2003 �յ�������
	virtual void OnFrontDisconnected(int nReason);

	///��¼������Ӧ
	virtual void OnRspUserLogin(CThostFtdcRspUserLoginField *pRspUserLogin, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�ǳ�������Ӧ
	virtual void OnRspUserLogout(CThostFtdcUserLogoutField *pUserLogout, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	//�г�״̬֪ͨ
	virtual void OnNtyMktStatus(CThostFtdcMarketStatusField *pfldMktStatus);

	///��Լ����״̬֪ͨ
	virtual void OnRtnInstrumentStatus(CThostFtdcInstrumentStatusField *pInstrumentStatus);

	///�����ѯ��Լ��Ӧ
	virtual void OnRspQryInstrument(CThostFtdcInstrumentField *pInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	//�����ѯ����Ʒ����Ӧ
	virtual void OnRspReqQryVarietyCode(CThostFtdcRspVarietyCodeField *pVarietyCode, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///����¼��������Ӧ
	virtual void OnRspOrderInsert(CThostFtdcRspInputOrderField *pRspInputOrder, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF�Ϲ�������Ӧ
	virtual void OnRspETFSubscriptionOrderInsert(CThostFtdcETFTradeDetailField *pETFSubscriptionOrderInsert, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF�깺������Ӧ
	virtual void OnRspETFPurchaseOrderInsert(CThostFtdcETFTradeDetailField *pETFSubscriptionOrderInsert, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF���������Ӧ
	virtual void OnRspETFRedeemInsert(CThostFtdcETFTradeDetailField *pETFSubscriptionOrderInsert, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF�˻���������Ӧ
	virtual void OnRspETFAccountBinding(CThostFtdcETFBindingStatusField *pETFAccountBinding, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF�˻����������Ӧ
	virtual void OnRspETFAccountUnbinding(CThostFtdcETFBindingStatusField *pETFAccountUnbinding, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///����֪ͨ
	virtual void OnRtnOrder(CThostFtdcOrderField *pOrder);

	///ǿ��֪ͨ
	virtual void OnForceLogout(CThostFtdcUserLogoutField *pLogout);

	//ETF�˻��󶨽��״̬֪ͨ
	virtual void OnRtnETFAccountBindingStatus(CThostFtdcETFBindingStatusField * pETFBindgingStatus);

	//ETF�����걨��״̬֪ͨ
	virtual void OnRtnETFOrder(CThostFtdcETFTradeDetailField *pEtfTradeDetail);

	///����Ӧ��
	virtual void OnRspOrderAction(CThostFtdcRspInputOrderActionField *pRspInputOrderAction, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///����Ӧ��
	virtual void OnRspError(CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�ɽ�֪ͨ
	virtual void OnRtnTrade(CThostFtdcTradeField *pTrade);

	///�����ѯ�ʽ��˻���Ӧ
	virtual void OnRspQryTradingAccount(CThostFtdcTradingAccountField *pTradingAccount, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///������ʷ�ʽ��ѯ
	virtual void OnRspQryHisCapital(CThostFtdcRspHisCapitalField *pHisCapital, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�����ѯ������Ӧ
	virtual void OnRspQryOrder(CThostFtdcOrderField *pOrder, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�����ѯ�ɽ���Ӧ
	virtual void OnRspQryTrade(CThostFtdcTradeField *pTrade, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�����ѯͶ���ֲ߳���Ӧ
	virtual void OnRspQryInvestorPosition(CThostFtdcInvestorPositionField *pInvestorPosition, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///��ѯ�ͻ������Ӧ
	virtual void OnRspQryClientStorage(CThostFtdcStorageField *pStorage, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///����\��֤���ʲ�ѯ��Ӧ
	virtual void OnRspQryCostMarginFeeRate(CThostFtdcRspCostMarginFeeField *pCostMarginFee, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///������ί����Ӧ
	virtual void OnRspConditionOrderInsert(CThostFtdcRspConditionOrderField *pConditionOrder, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///������������Ӧ
	virtual void OnRspConditionOrderAction(CThostFtdcRspConditionActionOrderField *pConditionActionOrder, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///������ί�в�ѯ��Ӧ
	virtual void OnRspQryConditionOrder(CThostFtdcRspConditionOrderQryField *pConditionOrderQry, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�������ɽ���ѯ��Ӧ
	virtual void OnRspQryConditionOrderTrade(CThostFtdcRspConditionOrderMatchField *pConditionOrderMatch, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�ͻ��Ự��Ϣͨ�ò�ѯ��Ӧ
	virtual void OnRspQryClientSessionInfo(CThostFtdcRspClientSessionField *pClientSessionField, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///��ѯ������Ϣ��Ӧ
	virtual void OnRspQryQuotation(CThostFtdcDepthMarketDataField *pDepthMarketData, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///��ѯͶ���ֲ߳���ϸ��Ӧ
	virtual void OnRspQryInvestorPositionDetail(CThostFtdcInvestorPositionDetailField *pInvestorPositionDetail, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF�����꽻�ײ�ѯ��Ӧ
	virtual void OnRspQryETFradeDetail(CThostFtdcETFTradeDetailField *pQryETFTradeDetail, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ETF�������嵥��ѯ
	virtual void OnRspQryETFPcfDetail(CThostFtdcETFPcfDetailField *pQryETFpcfDetail, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�޸�������Ӧ
	virtual void OnRspModifyPassword(CThostFtdcModifyPasswordRsqField *pRsqModifyPassword, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///���г������Ӧ
	virtual void OnRspB0CMoneyIO(CThostFtdcBOCMoneyIORspField *pRspBOCMoneyIO, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);



    //-------------------------------------------------------------------------------------
    //task������
    //-------------------------------------------------------------------------------------
    void processTask();

	void processFrontConnected(Task *task);

	void processFrontDisconnected(Task *task);

	void processRspUserLogin(Task *task);

	void processRspUserLogout(Task *task);

	void processNtyMktStatus(Task *task);

	void processRtnInstrumentStatus(Task *task);

	void processRspQryInstrument(Task *task);

	void processRspReqQryVarietyCode(Task *task);

	void processRspOrderInsert(Task *task);

	void processRspETFSubscriptionOrderInsert(Task *task);

	void processRspETFPurchaseOrderInsert(Task *task);

	void processRspETFRedeemInsert(Task *task);

	void processRspETFAccountBinding(Task *task);

	void processRspETFAccountUnbinding(Task *task);

	void processRtnOrder(Task *task);

	void processForceLogout(Task *task);

	void processRtnETFAccountBindingStatus(Task *task);

	void processRtnETFOrder(Task *task);

	void processRspOrderAction(Task *task);

	void processRspError(Task *task);

	void processRtnTrade(Task *task);

	void processRspQryTradingAccount(Task *task);

	void processRspQryHisCapital(Task *task);

	void processRspQryOrder(Task *task);

	void processRspQryTrade(Task *task);

	void processRspQryInvestorPosition(Task *task);

	void processRspQryClientStorage(Task *task);

	void processRspQryCostMarginFeeRate(Task *task);

	void processRspConditionOrderInsert(Task *task);

	void processRspConditionOrderAction(Task *task);

	void processRspQryConditionOrder(Task *task);

	void processRspQryConditionOrderTrade(Task *task);

	void processRspQryClientSessionInfo(Task *task);

	void processRspQryQuotation(Task *task);

	void processRspQryInvestorPositionDetail(Task *task);

	void processRspQryETFradeDetail(Task *task);

	void processRspQryETFPcfDetail(Task *task);

	void processRspModifyPassword(Task *task);

	void processRspB0CMoneyIO(Task *task);



    //-------------------------------------------------------------------------------------
    //data���ص������������ֵ�
    //error���ص������Ĵ����ֵ�
    //id������id
    //last���Ƿ�Ϊ��󷵻�
    //i������
    //-------------------------------------------------------------------------------------    

	virtual void onFrontConnected(int nResult) {};

	virtual void onFrontDisconnected(int nReason) {};

	virtual void onRspUserLogin(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspUserLogout(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onNtyMktStatus(const dict &data) {};

	virtual void onRtnInstrumentStatus(const dict &data) {};

	virtual void onRspQryInstrument(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspReqQryVarietyCode(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspOrderInsert(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspETFSubscriptionOrderInsert(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspETFPurchaseOrderInsert(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspETFRedeemInsert(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspETFAccountBinding(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspETFAccountUnbinding(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRtnOrder(const dict &data) {};

	virtual void onForceLogout(const dict &data) {};

	virtual void onRtnETFAccountBindingStatus(const dict &data) {};

	virtual void onRtnETFOrder(const dict &data) {};

	virtual void onRspOrderAction(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspError(const dict &error, int reqid, bool last) {};

	virtual void onRtnTrade(const dict &data) {};

	virtual void onRspQryTradingAccount(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryHisCapital(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryOrder(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryTrade(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryInvestorPosition(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryClientStorage(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryCostMarginFeeRate(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspConditionOrderInsert(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspConditionOrderAction(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryConditionOrder(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryConditionOrderTrade(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryClientSessionInfo(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryQuotation(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryInvestorPositionDetail(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryETFradeDetail(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspQryETFPcfDetail(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspModifyPassword(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspB0CMoneyIO(const dict &data, const dict &error, int reqid, bool last) {};


    //-------------------------------------------------------------------------------------
    //req:���������������ֵ�
    //-------------------------------------------------------------------------------------
	void createGoldTraderApi(string pszFlowPath = " ");

	int exit();

	void release();

	bool init();

	void registerFront(string pszFrontAddress);

	void subscribePrivateTopic(int nResumeType);

	void subscribePublicTopic(int nResumeType);


	int join();

	int reqUserLogin(const dict &req, int reqid);

	int reqUserLogout(const dict &req, int reqid);

	int reqQryInstrument(const dict &req, int reqid);

	int reqQryVarietyCode(const dict &req, int reqid);

	int reqOrderInsert(const dict &req, int reqid);

	int reqOrderAction(const dict &req, int reqid);

	int reqQryInvestorPosition(const dict &req, int reqid);

	int reqQryTradingAccount(const dict &req, int reqid);

	int reqQryTrade(const dict &req, int reqid);

	int reqQryOrder(const dict &req, int reqid);

	int reqQryStorage(const dict &req, int reqid);

	int reqQryCostMarginFeeRate(const dict &req, int reqid);

	int reqConditionOrderInsert(const dict &req, int reqid);

	int reqConditionOrderAction(const dict &req, int reqid);

	int reqQryConditionOrder(const dict &req, int reqid);

	int reqQryConditionOrderTrade(const dict &req, int reqid);

	int reqQryClientSessionInfo(const dict &req, int reqid);

	int reqQryQuotation(const dict &req, int reqid);

	int reqQryInvestorPositionDetail(const dict &req, int reqid);

	int reqModifyPassword(const dict &req, int reqid);

	int reqQryHisCapital(const dict &req, int reqid);

	int reqETFSubScription(const dict &req, int reqid);

	int reqETFApplyForPurchase(const dict &req, int reqid);

	int reqETFRedeem(const dict &req, int reqid);

	int reqETFAccountBinding(const dict &req, int reqid);

	int reqETFAccountUnbinding(const dict &req, int reqid);

	int reqETFTradeDetail(const dict &req, int reqid);

	int reqETFPcfDetail(const dict &req, int reqid);

	int reqBOCMoneyIO(const dict &req, int reqid);


};
