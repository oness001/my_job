//ϵͳ
#ifdef WIN32
#include "stdafx.h"
#endif

#include "vnxtp.h"
#include "pybind11/pybind11.h"
#include "xtp/xtp_quote_api.h"


using namespace pybind11;
using namespace XTP::API;


//����
#define ONDISCONNECTED 0
#define ONERROR 1
#define ONSUBMARKETDATA 2
#define ONUNSUBMARKETDATA 3
#define ONDEPTHMARKETDATA 4
#define ONSUBORDERBOOK 5
#define ONUNSUBORDERBOOK 6
#define ONORDERBOOK 7
#define ONSUBTICKBYTICK 8
#define ONUNSUBTICKBYTICK 9
#define ONTICKBYTICK 10
#define ONSUBSCRIBEALLMARKETDATA 11
#define ONUNSUBSCRIBEALLMARKETDATA 12
#define ONSUBSCRIBEALLORDERBOOK 13
#define ONUNSUBSCRIBEALLORDERBOOK 14
#define ONSUBSCRIBEALLTICKBYTICK 15
#define ONUNSUBSCRIBEALLTICKBYTICK 16
#define ONQUERYALLTICKERS 17
#define ONQUERYTICKERSPRICEINFO 18
#define ONSUBSCRIBEALLOPTIONMARKETDATA 19
#define ONUNSUBSCRIBEALLOPTIONMARKETDATA 20
#define ONSUBSCRIBEALLOPTIONORDERBOOK 21
#define ONUNSUBSCRIBEALLOPTIONORDERBOOK 22
#define ONSUBSCRIBEALLOPTIONTICKBYTICK 23
#define ONUNSUBSCRIBEALLOPTIONTICKBYTICK 24



///-------------------------------------------------------------------------------------
///C++ SPI�Ļص���������ʵ��
///-------------------------------------------------------------------------------------

//API�ļ̳�ʵ��
class MdApi : public QuoteSpi
{
private:
	QuoteApi* api;				//API����
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

	///���ͻ����������̨ͨ�����ӶϿ�ʱ���÷��������á�
	///@param reason ����ԭ��������������Ӧ
	///@remark api�����Զ������������߷���ʱ�����û�����ѡ����������������ڴ˺����е���Login���µ�¼��ע���û����µ�¼����Ҫ���¶�������
	virtual void OnDisconnected(int reason);


	///����Ӧ��
	///@param error_info ����������Ӧ��������ʱ�ľ���Ĵ������ʹ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark �˺���ֻ���ڷ�������������ʱ�Ż���ã�һ�������û�����
	virtual void OnError(XTPRI *error_info);

	///��������Ӧ�𣬰�����Ʊ��ָ������Ȩ
	///@param ticker ��ϸ�ĺ�Լ�������
	///@param error_info ���ĺ�Լ��������ʱ�Ĵ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴ζ��ĵ����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	///@remark ÿ�����ĵĺ�Լ����Ӧһ������Ӧ����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnSubMarketData(XTPST *ticker, XTPRI *error_info, bool is_last);

	///�˶�����Ӧ�𣬰�����Ʊ��ָ������Ȩ
	///@param ticker ��ϸ�ĺ�Լȡ���������
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴�ȡ�����ĵ����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	///@remark ÿ��ȡ�����ĵĺ�Լ����Ӧһ��ȡ������Ӧ����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnUnSubMarketData(XTPST *ticker, XTPRI *error_info, bool is_last);

	///�������֪ͨ��������һ��һ����
	///@param market_data ��������
	///@param bid1_qty ��һ��������
	///@param bid1_count ��һ���е���Чί�б���
	///@param max_bid1_count ��һ������ί�б���
	///@param ask1_qty ��һ��������
	///@param ask1_count ��һ���е���Чί�б���
	///@param max_ask1_count ��һ������ί�б���
	///@remark ��Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnDepthMarketData(XTPMD *market_data, int64_t bid1_qty[], int32_t bid1_count, int32_t max_bid1_count, int64_t ask1_qty[], int32_t ask1_count, int32_t max_ask1_count);

	///�������鶩����Ӧ�𣬰�����Ʊ��ָ������Ȩ
	///@param ticker ��ϸ�ĺ�Լ�������
	///@param error_info ���ĺ�Լ��������ʱ�Ĵ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴ζ��ĵ����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	///@remark ÿ�����ĵĺ�Լ����Ӧһ������Ӧ����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnSubOrderBook(XTPST *ticker, XTPRI *error_info, bool is_last);

	///�˶����鶩����Ӧ�𣬰�����Ʊ��ָ������Ȩ
	///@param ticker ��ϸ�ĺ�Լȡ���������
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴�ȡ�����ĵ����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	///@remark ÿ��ȡ�����ĵĺ�Լ����Ӧһ��ȡ������Ӧ����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnUnSubOrderBook(XTPST *ticker, XTPRI *error_info, bool is_last);

	///���鶩����֪ͨ��������Ʊ��ָ������Ȩ
	///@param order_book ���鶩�������ݣ���Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnOrderBook(XTPOB *order_book);

	///�����������Ӧ�𣬰�����Ʊ��ָ������Ȩ
	///@param ticker ��ϸ�ĺ�Լ�������
	///@param error_info ���ĺ�Լ��������ʱ�Ĵ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴ζ��ĵ����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	///@remark ÿ�����ĵĺ�Լ����Ӧһ������Ӧ����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnSubTickByTick(XTPST *ticker, XTPRI *error_info, bool is_last);

	///�˶��������Ӧ�𣬰�����Ʊ��ָ������Ȩ
	///@param ticker ��ϸ�ĺ�Լȡ���������
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴�ȡ�����ĵ����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	///@remark ÿ��ȡ�����ĵĺ�Լ����Ӧһ��ȡ������Ӧ����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnUnSubTickByTick(XTPST *ticker, XTPRI *error_info, bool is_last);

	///�������֪ͨ��������Ʊ��ָ������Ȩ
	///@param tbt_data ����������ݣ��������ί�к���ʳɽ�����Ϊ���ýṹ�壬��Ҫ����type�����������ί�л�����ʳɽ�����Ҫ���ٷ��أ���������������Ϣ������������ʱ���ᴥ������
	virtual void OnTickByTick(XTPTBT *tbt_data);

	///����ȫ�г��Ĺ�Ʊ����Ӧ��
	///@param exchange_id ��ʾ��ǰȫ���ĵ��г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnSubscribeAllMarketData(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///�˶�ȫ�г��Ĺ�Ʊ����Ӧ��
	///@param exchange_id ��ʾ��ǰ�˶����г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnUnSubscribeAllMarketData(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///����ȫ�г��Ĺ�Ʊ���鶩����Ӧ��
	///@param exchange_id ��ʾ��ǰȫ���ĵ��г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnSubscribeAllOrderBook(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///�˶�ȫ�г��Ĺ�Ʊ���鶩����Ӧ��
	///@param exchange_id ��ʾ��ǰ�˶����г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnUnSubscribeAllOrderBook(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///����ȫ�г��Ĺ�Ʊ�������Ӧ��
	///@param exchange_id ��ʾ��ǰȫ���ĵ��г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnSubscribeAllTickByTick(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///�˶�ȫ�г��Ĺ�Ʊ�������Ӧ��
	///@param exchange_id ��ʾ��ǰ�˶����г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnUnSubscribeAllTickByTick(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);


	///��ѯ�ɽ��׺�Լ��Ӧ��
	///@param ticker_info �ɽ��׺�Լ��Ϣ
	///@param error_info ��ѯ�ɽ��׺�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴β�ѯ�ɽ��׺�Լ�����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	virtual void OnQueryAllTickers(XTPQSI* ticker_info, XTPRI *error_info, bool is_last);

	///��ѯ��Լ�����¼۸���ϢӦ��
	///@param ticker_info ��Լ�����¼۸���Ϣ
	///@param error_info ��ѯ��Լ�����¼۸���Ϣʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@param is_last �Ƿ�˴β�ѯ�����һ��Ӧ�𣬵�Ϊ���һ����ʱ��Ϊtrue�����Ϊfalse����ʾ��������������Ϣ��Ӧ
	virtual void OnQueryTickersPriceInfo(XTPTPI* ticker_info, XTPRI *error_info, bool is_last);

	///����ȫ�г�����Ȩ����Ӧ��
	///@param exchange_id ��ʾ��ǰȫ���ĵ��г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnSubscribeAllOptionMarketData(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///�˶�ȫ�г�����Ȩ����Ӧ��
	///@param exchange_id ��ʾ��ǰ�˶����г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnUnSubscribeAllOptionMarketData(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///����ȫ�г�����Ȩ���鶩����Ӧ��
	///@param exchange_id ��ʾ��ǰȫ���ĵ��г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnSubscribeAllOptionOrderBook(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///�˶�ȫ�г�����Ȩ���鶩����Ӧ��
	///@param exchange_id ��ʾ��ǰ�˶����г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnUnSubscribeAllOptionOrderBook(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///����ȫ�г�����Ȩ�������Ӧ��
	///@param exchange_id ��ʾ��ǰȫ���ĵ��г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnSubscribeAllOptionTickByTick(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);

	///�˶�ȫ�г�����Ȩ�������Ӧ��
	///@param exchange_id ��ʾ��ǰ�˶����г������ΪXTP_EXCHANGE_UNKNOWN����ʾ����ȫ�г���XTP_EXCHANGE_SH��ʾΪ�Ϻ�ȫ�г���XTP_EXCHANGE_SZ��ʾΪ����ȫ�г�
	///@param error_info ȡ�����ĺ�Լʱ��������ʱ���صĴ�����Ϣ����error_infoΪ�գ�����error_info.error_idΪ0ʱ������û�д���
	///@remark ��Ҫ���ٷ���
	virtual void OnUnSubscribeAllOptionTickByTick(XTP_EXCHANGE_TYPE exchange_id, XTPRI *error_info);
	//-------------------------------------------------------------------------------------
	//task������
	//-------------------------------------------------------------------------------------

	void processTask();

	void processDisconnected(Task *task);

	void processError(Task *task);

	void processSubMarketData(Task *task);

	void processUnSubMarketData(Task *task);

	void processDepthMarketData(Task *task);

	void processSubOrderBook(Task *task);

	void processUnSubOrderBook(Task *task);

	void processOrderBook(Task *task);

	void processSubTickByTick(Task *task);

	void processUnSubTickByTick(Task *task);

	void processTickByTick(Task *task);

	void processSubscribeAllMarketData(Task *task);

	void processUnSubscribeAllMarketData(Task *task);

	void processSubscribeAllOrderBook(Task *task);

	void processUnSubscribeAllOrderBook(Task *task);

	void processSubscribeAllTickByTick(Task *task);

	void processUnSubscribeAllTickByTick(Task *task);

	void processQueryAllTickers(Task *task);

	void processQueryTickersPriceInfo(Task *task);

	void processSubscribeAllOptionMarketData(Task *task);

	void processUnSubscribeAllOptionMarketData(Task *task);

	void processSubscribeAllOptionOrderBook(Task *task);

	void processUnSubscribeAllOptionOrderBook(Task *task);

	void processSubscribeAllOptionTickByTick(Task *task);

	void processUnSubscribeAllOptionTickByTick(Task *task);


	//-------------------------------------------------------------------------------------
	//data���ص������������ֵ�
	//error���ص������Ĵ����ֵ�
	//id������id
	//last���Ƿ�Ϊ��󷵻�
	//i������
	//-------------------------------------------------------------------------------------

	virtual void onDisconnected(int reqid) {};

	virtual void onError(const dict &error) {};

	virtual void onSubMarketData(const dict &data, const dict &error, bool last) {};

	virtual void onUnSubMarketData(const dict &data, const dict &error, bool last) {};

	virtual void onDepthMarketData(const dict &data) {};

	virtual void onSubOrderBook(const dict &data, const dict &error, bool last) {};

	virtual void onUnSubOrderBook(const dict &data, const dict &error, bool last) {};

	virtual void onOrderBook(const dict &data) {};

	virtual void onSubTickByTick(const dict &data, const dict &error, bool last) {};

	virtual void onUnSubTickByTick(const dict &data, const dict &error, bool last) {};

	virtual void onTickByTick(const dict &data) {};

	virtual void onSubscribeAllMarketData(int extra, const dict &error) {};

	virtual void onUnSubscribeAllMarketData(int extra, const dict &error) {};

	virtual void onSubscribeAllOrderBook(int extra, const dict &error) {};

	virtual void onUnSubscribeAllOrderBook(int extra, const dict &error) {};

	virtual void onSubscribeAllTickByTick(int extra, const dict &error) {};

	virtual void onUnSubscribeAllTickByTick(int extra, const dict &error) {};

	virtual void onQueryAllTickers(const dict &data, const dict &error, bool last) {};

	virtual void onQueryTickersPriceInfo(const dict &data, const dict &error, bool last) {};

	virtual void onSubscribeAllOptionMarketData(int extra, const dict &error) {};

	virtual void onUnSubscribeAllOptionMarketData(int extra, const dict &error) {};

	virtual void onSubscribeAllOptionOrderBook(int extra, const dict &error) {};

	virtual void onUnSubscribeAllOptionOrderBook(int extra, const dict &error) {};

	virtual void onSubscribeAllOptionTickByTick(int extra, const dict &error) {};

	virtual void onUnSubscribeAllOptionTickByTick(int extra, const dict &error) {};



	//-------------------------------------------------------------------------------------
	//req:���������������ֵ�
	//-------------------------------------------------------------------------------------

	void createQuoteApi(int client_id, string save_file_path);

	void release();

	void init();

	int exit();

	string getTradingDay();

	string getApiVersion();

	dict getApiLastError();

	void setUDPBufferSize(int buff_size);

	void setHeartBeatInterval(int interval);

	int subscribeMarketData(string ticker, int count, int exchange_id);

	int unSubscribeMarketData(string ticker, int count, int exchange_id);

	int subscribeOrderBook(string ticker, int count, int exchange_id);

	int unSubscribeOrderBook(string ticker, int count, int exchange_id);

	int subscribeTickByTick(string ticker, int count, int exchange_id);

	int unSubscribeTickByTick(string ticker, int count, int exchange_id);

	int subscribeAllMarketData(int exchange_id);

	int unSubscribeAllMarketData(int exchange_id);

	int subscribeAllOrderBook(int exchange_id);

	int unSubscribeAllOrderBook(int exchange_id);

	int subscribeAllTickByTick(int exchange_id);

	int unSubscribeAllTickByTick(int exchange_id);

	int login(string ip, int port, string user, string password, int sock_type);

	int logout();

	int queryAllTickers(int exchange_id);

	int queryTickersPriceInfo(string ticker, int count, int exchange_id);

	int queryAllTickersPriceInfo();


};
