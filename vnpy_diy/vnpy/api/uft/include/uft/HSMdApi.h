/**
* @author �ٺط�
* @company �������ӹɷ����޹�˾
* @file HSMdApi.h
* @brief ����������ӿ�
* @date    20170927
*/
#ifndef _HS_MD_API_H_
#define _HS_MD_API_H_

#include "HSStruct.h"


#ifdef WIN32
    #define MD_API_EXPORT 
#else
    #ifdef HSMDAPI_EXPORTS
        #define MD_API_EXPORT __attribute__((visibility("default"))) 
    #else
        #define MD_API_EXPORT
    #endif
#endif

///�ص�����
class CHSMdSpi
{
public:
    /// Description: ���ͻ����뽻�׺�̨��ʼ����ͨ�����ӣ����ӳɹ���˷������ص���
    virtual void OnFrontConnected(){};

    /// Description:���ͻ����뽻�׺�̨ͨ�������쳣ʱ���÷��������á�
    /// Others     :ͨ��GetApiErrorMsg(nResult)��ȡ��ϸ������Ϣ��
    virtual void OnFrontDisconnected(int nResult){};

    /// Description: ����-����Ӧ��
    virtual void OnRspDepthMarketDataSubscribe(CHSRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};

    /// Description: ����ȡ��-����Ӧ��
    virtual void OnRspDepthMarketDataCancel(CHSRspInfoField *pRspInfo, int nRequestID, bool bIsLast) {};

    /// Description: ����-����
    virtual void OnRtnDepthMarketData(CHSDepthMarketDataField *pDepthMarketData) {};

};

///����
class CHSMdApi
{
public:
    /// Description: ɾ���ӿڶ�����
    virtual void ReleaseApi() = 0;

    /// Description: ��ʼ������
    ///              pszLicFile            ͨѶ֤��
    ///              pszSafeLevel          ��ȫ����
    ///              pszPwd                ͨѶ����
    ///              pszSslFile            SSL֤��
    ///              pszSslPwd             SSL����
    virtual int Init(const char *pszLicFile,const char *pszSafeLevel = "", const char *pszPwd = "", const char *pszSslFile = "", const char *pszSslPwd = "") = 0;

    /// Description: API�ͽ��׹�̨�������ӣ����ӽ����ɹ������̻߳�ȴ��û��������߳��˳�
    /// Return     ��int 0��ʾ���ӽ����ɹ��������ʾʧ�ܣ�ͨ������GetApiErrorMsg���Ի�ȡ��ϸ������Ϣ
    virtual int Join() = 0;

    /// Description: ע��ǰ�û������ַ
    /// Input      : pszFrontAddress            ǰ�û������ַ  �磺��tcp://127.0.0.1:17001��
    /// Return     : int 0��ʾ���óɹ��������ʾʧ�ܣ�ͨ������GetApiErrorMsg���Ի�ȡ��ϸ������Ϣ
    virtual int RegisterFront(const char *pszFrontAddress) = 0;

    /// Description: ע��Fens�����ַ
    /// Input      : pszFensAddress            Fens�����ַ
    /// Input      : pszAccountID              �˺�
    /// Return     : int 0��ʾ���óɹ��������ʾʧ�ܣ�ͨ������GetApiErrorMsg���Ի�ȡ��ϸ������Ϣ
    virtual int RegisterFensServer(const char *pszFensAddress, const char *pszAccountID) = 0;

    /// Description: ע��ص��ӿ�
    /// Input      : pSpi            �����Իص��ӿ����ʵ��     
    virtual void RegisterSpi(CHSMdSpi *pSpi) = 0;

    /// Description: ����-��������
    /// Input      : pReqDepthMarketDataSubscribe[]        ���鶩������ṹ������ 
    ///              nCount                                ���������Լ����
    ///              nRequestID                            ������ 
    virtual int ReqDepthMarketDataSubscribe(CHSReqDepthMarketDataField pReqDepthMarketDataSubscribe[], int nCount, int nRequestID) = 0;

    /// Description: ����ȡ��-��������
    /// Input      : pReqDepthMarketDataCancel[]        ���鶩������ṹ������ 
    ///              nCount                             ���������Լ����
    ///              nRequestID                         ������ 
    virtual int ReqDepthMarketDataCancel(CHSReqDepthMarketDataField pReqDepthMarketDataCancel[], int nCount, int nRequestID) = 0;

    ///////////////////////////////////////////////////////////////////////////////////////
    /// Description: ��ô�����ϸ��Ϣ
    /// Input      : nErrorCode            �����
    /// Return     ��������Ϣ
    ///////////////////////////////////////////////////////////////////////////////////////
    virtual const char* GetApiErrorMsg(int nErrorCode) = 0;

protected:
    ~CHSMdApi(){};
 };
#ifdef __cplusplus
extern "C"
{
#endif
    /// Description: ��ȡAPI�汾��
    /// Return     : API�汾��
    const char* GetMdApiVersion();

    /// Description: ��������ӿ�
    /// Input      : pszFlowPath    ��־·��   
    CHSMdApi* NewMdApi(const char *pszFlowPath);

#ifdef __cplusplus
}
#endif
#endif
