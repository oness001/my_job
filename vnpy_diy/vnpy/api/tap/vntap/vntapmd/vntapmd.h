//ϵͳ
#ifdef WIN32
#include "stdafx.h"
#endif

#include "vntap.h"
#include "pybind11/pybind11.h"
#include "tap/TapQuoteAPI.h"


using namespace pybind11;

//����
#define ONRSPLOGIN 0
#define ONAPIREADY 1
#define ONDISCONNECT 2
#define ONRSPQRYCOMMODITY 3
#define ONRSPQRYCONTRACT 4
#define ONRSPSUBSCRIBEQUOTE 5
#define ONRSPUNSUBSCRIBEQUOTE 6
#define ONRTNQUOTE 7


///-------------------------------------------------------------------------------------
///C++ SPI�Ļص���������ʵ��
///-------------------------------------------------------------------------------------

//API�ļ̳�ʵ��
class MdApi : public ITapQuoteAPINotify
{
private:
	ITapQuoteAPI* api;				//API����
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
	/**
	* @brief	ϵͳ��¼���̻ص���
	* @details	�˺���ΪLogin()��¼�����Ļص�������Login()�ɹ���������·���ӣ�Ȼ��API������������͵�¼��֤��Ϣ��
	*			��¼�ڼ�����ݷ�������͵�¼�Ļ�����Ϣ���ݵ��˻ص������С�
	* @param[in] errorCode ���ش�����,0��ʾ�ɹ���
	* @param[in] info ��½Ӧ����Ϣ�����errorCode!=0����info=NULL��
	* @attention	�ûص����سɹ���˵���û���¼�ɹ������ǲ�����API׼����ϡ���Ҫ�ȵ�OnAPIReady���ܽ��в�ѯ�붩������
	* @ingroup G_Q_Login
	*/
	virtual void TAP_CDECL OnRspLogin(TAPIINT32 errorCode, const TapAPIQuotLoginRspInfo *info);
	/**
	* @brief	֪ͨ�û�API׼��������
	* @details	ֻ���û��ص��յ��˾���֪ͨʱ���ܽ��к����ĸ����������ݲ�ѯ������\n
	*			�˻ص�������API�ܷ����������ı�־��
	* @attention  ������ſ��Խ��к�����������
	* @ingroup G_Q_Login
	*/
	virtual void TAP_CDECL OnAPIReady();
	/**
	* @brief	API�ͷ���ʧȥ���ӵĻص�
	* @details	��APIʹ�ù������������߱��������������ʧȥ���Ӻ󶼻ᴥ���˻ص�֪ͨ�û���������������Ѿ��Ͽ���
	* @param[in] reasonCode �Ͽ�ԭ����롣����ԭ����μ��������б� \n
	* @ingroup G_Q_Disconnect
	*/
	virtual void TAP_CDECL OnDisconnect(TAPIINT32 reasonCode);
	/**
	* @brief	��������Ʒ����Ϣ��
	* @details	�˻ص��ӿ��������û����صõ�������Ʒ����Ϣ��
	* @param[in] sessionID ����ĻỰID
	* @param[in] errorCode �����룬��errorCode!=0ʱ,infoΪNULL��
	* @param[in] isLast ��ʾ�Ƿ������һ�����ݣ�
	* @param[in] info ���ص���Ϣ�������ʼָ�롣
	* @attention  ��Ҫ�޸ĺ�ɾ��info��ָʾ�����ݣ��������ý���������������Ч��
	* @ingroup G_Q_Commodity
	*/
	virtual void TAP_CDECL OnRspQryCommodity(TAPIUINT32 sessionID, TAPIINT32 errorCode, TAPIYNFLAG isLast, const TapAPIQuoteCommodityInfo *info);
	/**
	* @brief ����ϵͳ�к�Լ��Ϣ
	* @param[in] sessionID ����ĻỰID��
	* @param[in] errorCode �����룬��errorCode!=0ʱ,infoΪNULL��
	* @param[in] isLast ��ʾ�Ƿ������һ�����ݣ�
	* @param[in] info		ָ�򷵻ص���Ϣ�ṹ�塣��errorCode��Ϊ0ʱ��infoΪ�ա�
	* @attention ��Ҫ�޸ĺ�ɾ��info��ָʾ�����ݣ��������ý���������������Ч��
	* @ingroup G_Q_Contract
	*/
	virtual void TAP_CDECL OnRspQryContract(TAPIUINT32 sessionID, TAPIINT32 errorCode, TAPIYNFLAG isLast, const TapAPIQuoteContractInfo *info);
	/**
	* @brief	���ض��������ȫ�ġ�
	* @details	�˻ص��ӿ��������ض��������ȫ�ġ�ȫ��Ϊ��ǰʱ���������Ϣ��
	* @param[in] sessionID ����ĻỰID��
	* @param[in] isLast ��ʾ�Ƿ������һ�����ݣ�
	* @param[in] errorCode �����룬��errorCode!=0ʱ,infoΪNULL��
	* @param[in] info		ָ�򷵻ص���Ϣ�ṹ�塣��errorCode��Ϊ0ʱ��infoΪ�ա�
	* @attention  ��Ҫ�޸ĺ�ɾ��info��ָʾ�����ݣ��������ý���������������Ч��
	* @ingroup G_Q_Quote
	*/
	virtual void TAP_CDECL OnRspSubscribeQuote(TAPIUINT32 sessionID, TAPIINT32 errorCode, TAPIYNFLAG isLast, const TapAPIQuoteWhole *info);
	/**
	* @brief �˶�ָ����Լ������Ľ���ص�
	* @param[in] sessionID ����ĻỰID��
	* @param[in] errorCode �����룬��errorCode!=0ʱ,infoΪNULL��
	* @param[in] isLast ��ʾ�Ƿ������һ�����ݣ�
	* @param[in] info		ָ�򷵻ص���Ϣ�ṹ�塣��errorCode��Ϊ0ʱ��infoΪ�ա�
	* @attention  ��Ҫ�޸ĺ�ɾ��info��ָʾ�����ݣ��������ý���������������Ч��
	* @ingroup G_Q_Quote
	*/
	virtual void TAP_CDECL OnRspUnSubscribeQuote(TAPIUINT32 sessionID, TAPIINT32 errorCode, TAPIYNFLAG isLast, const TapAPIContract *info);
	/**
	* @brief	���ض�������ı仯���ݡ�
	* @details	�˻ص��ӿ�����֪ͨ�û�������Ϣ�����˱仯�������û��ύ�µ�����ȫ�ġ�
	* @param[in] info ���µ�����ȫ������
	* @attention ��Ҫ�޸ĺ�ɾ��Quoteָʾ�����ݣ��������ý���������������Ч��
	* @ingroup G_Q_Quote
	*/
	virtual void TAP_CDECL OnRtnQuote(const TapAPIQuoteWhole *info);

	//-------------------------------------------------------------------------------------
	//task������
	//-------------------------------------------------------------------------------------

	void processTask();

	void processRspLogin(Task *task);

	void processAPIReady(Task *task);

	void processDisconnect(Task *task);

	void processRspQryCommodity(Task *task);

	void processRspQryContract(Task *task);

	void processRspSubscribeQuote(Task *task);

	void processRspUnSubscribeQuote(Task *task);

	void processRtnQuote(Task *task);


	//-------------------------------------------------------------------------------------
	//data���ص������������ֵ�
	//error���ص������Ĵ����ֵ�
	//id������id
	//last���Ƿ�Ϊ��󷵻�
	//i������
	//-------------------------------------------------------------------------------------

	virtual void onRspLogin(int error, const dict &data) {};

	virtual void onAPIReady() {};

	virtual void onDisconnect(int reason) {};

	virtual void onRspQryCommodity(unsigned int session, int error, char last, const dict &data) {};

	virtual void onRspQryContract(unsigned int session, int error, char last, const dict &data) {};

	virtual void onRspSubscribeQuote(unsigned int session, int error, char last, const dict &data) {};

	virtual void onRspUnSubscribeQuote(unsigned int session, int error, char last, const dict &data) {};

	virtual void onRtnQuote(const dict &data) {};

	//-------------------------------------------------------------------------------------
	//req:���������������ֵ�
	//-------------------------------------------------------------------------------------

	void createTapQuoteAPI(const dict &req, int &iResult);

	void release();

	void init();

	int exit();

	string getTapQuoteAPIVersion();

	int setTapQuoteAPIDataPath(string path);

	int setTapQuoteAPILogLevel(string level); //1

	int setHostAddress(string IP, int port); //2

	int login(const dict &req);

	int disconnect();

	int subscribeQuote(const dict &req); //3

	int qryCommodity();

	int qryContract(const dict &req);
};
