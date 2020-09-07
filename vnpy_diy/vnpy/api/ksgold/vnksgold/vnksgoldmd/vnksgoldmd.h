//ϵͳ
#ifdef WIN32
#include "stdafx.h"
#endif

#include "vnksgold.h"
#include "pybind11/pybind11.h"
#include "ksgold/KSGoldQuotMarketDataApi.h"


using namespace pybind11;
using namespace KSGoldTradeAPI;

//����
#define ONFRONTCONNECTED 0
#define ONFRONTDISCONNECTED 1
#define ONRSPUSERLOGIN 2
#define ONRSPUSERLOGOUT 3
#define ONRSPERROR 4
#define ONRSPSUBMARKETDATA 5
#define ONRSPUNSUBMARKETDATA 6
#define ONRTNDEPTHMARKETDATA 7



///-------------------------------------------------------------------------------------
///C++ SPI�Ļص���������ʵ��
///-------------------------------------------------------------------------------------

//API�ļ̳�ʵ��
class MdApi : public CKSGoldQuoSpi
{
private:
	CKSGoldQuotApi* api;				//API����
	thread task_thread;					//�����߳�ָ�루��python���������ݣ�
	TaskQueue task_queue;			    //�������
	bool active = false;				//����״̬

public:
	MdApi()
	{
	};

	~MdApi()
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

	///����Ӧ��
	virtual void OnRspError(CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///��������Ӧ��
	virtual void OnRspSubMarketData(CThostFtdcSpecificInstrumentField *pSpecificInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///ȡ����������Ӧ��
	virtual void OnRspUnSubMarketData(CThostFtdcSpecificInstrumentField *pSpecificInstrument, CThostFtdcRspInfoField *pRspInfo, int nRequestID, bool bIsLast);

	///�������֪ͨ
	virtual void OnRtnDepthMarketData(CThostFtdcDepthMarketDataField *pDepthMarketData);


	//-------------------------------------------------------------------------------------
	//task������
	//-------------------------------------------------------------------------------------

	void processTask();

	void processFrontConnected(Task *task);

	void processFrontDisconnected(Task *task);

	void processRspUserLogin(Task *task);

	void processRspUserLogout(Task *task);

	void processRspError(Task *task);

	void processRspSubMarketData(Task *task);

	void processRspUnSubMarketData(Task *task);

	void processRtnDepthMarketData(Task *task);
	//-------------------------------------------------------------------------------------
	//data���ص������������ֵ�
	//error���ص������Ĵ����ֵ�
	//id������id
	//last���Ƿ�Ϊ��󷵻�
	//i������
	//-------------------------------------------------------------------------------------
	virtual void onFrontConnected(int nResult) {};

	virtual void onFrontDisconnected(int nResult) {};

	virtual void onRspUserLogin(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspUserLogout(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspError(const dict &error, int reqid, bool last) {};

	virtual void onRspSubMarketData(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRspUnSubMarketData(const dict &data, const dict &error, int reqid, bool last) {};

	virtual void onRtnDepthMarketData(const dict &data) {};

	//-------------------------------------------------------------------------------------
	//req:���������������ֵ�
	//-------------------------------------------------------------------------------------

	void release();

	bool init();

	int exit();

	int join();

	void registerFront(string pszFrontAddress);

	void createGoldQutoApi(string pszFlowPath = " ");

	int subscribeMarketData(string ppInstrumentID, int reqid);

	int unSubscribeMarketData(string ppInstrumentID, int reqid);

	int reqUserLogin(const dict &req, int reqid);

	int reqUserLogout(const dict &req, int reqid);
};
